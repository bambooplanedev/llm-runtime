use crate::state::{Cluster, NodeState, NodeView, PROTO};
use futures_util::stream::{FuturesUnordered, StreamExt};
use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

pub type SharedCluster = Arc<RwLock<Cluster>>;

const SERVICE: &str = "_llmrt._tcp.local.";
/// Скільки пропущених poll-ів до dead (§3).
const MAX_MISSES: u32 = 3;
/// Таймаут запиту `/state` (§3).
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);
/// Не частіше одного warning про proto на вузол за хвилину (§7).
const PROTO_WARN_EVERY: Duration = Duration::from_secs(60);

/// Живий вузол опитуємо раз на 3 с, мертвий — раз на 10 с (§3).
pub fn poll_interval(alive: bool) -> Duration {
    if alive {
        Duration::from_secs(3)
    } else {
        Duration::from_secs(10)
    }
}

#[derive(Clone)]
pub struct Discovery {
    pub cluster: SharedCluster,
    self_id: String,
    self_port: u16,
    seeds: Vec<String>,
    /// node_id (з TXT) → адреси, які приніс mDNS
    resolved: Arc<Mutex<HashMap<String, HashSet<String>>>>,
}

impl Discovery {
    pub fn new(self_id: String, self_port: u16, peers: Vec<String>) -> Discovery {
        Discovery {
            cluster: Default::default(),
            self_id,
            self_port,
            seeds: peers,
            resolved: Default::default(),
        }
    }

    /// Вузол відповів: оновити стан за `node_id`, `addr` — та адреса, що реально відповіла (§4).
    pub fn ingest(cluster: &SharedCluster, st: NodeState, addr_used: &str) {
        let mut st = st;
        st.addr = addr_used.to_string();
        let mut g = cluster.write().unwrap();
        match g.get_mut(&st.node_id) {
            Some(v) => {
                v.state = st;
                v.alive = true;
                v.misses = 0;
                v.last_seen = Instant::now();
            }
            None => {
                g.insert(
                    st.node_id.clone(),
                    NodeView {
                        state: st,
                        alive: true,
                        last_seen: Instant::now(),
                        misses: 0,
                    },
                );
            }
        }
    }

    /// Пропущений poll; після `MAX_MISSES` підряд — dead.
    pub fn miss(cluster: &SharedCluster, node_id: &str) {
        let mut g = cluster.write().unwrap();
        if let Some(v) = g.get_mut(node_id) {
            v.misses += 1;
            if v.misses >= MAX_MISSES && v.alive {
                v.alive = false;
                tracing::warn!("node {node_id} dead after {MAX_MISSES} misses");
            }
        }
    }

    /// Негайний dead — з gateway, коли з'єднання відкинуто.
    pub fn mark_dead(cluster: &SharedCluster, node_id: &str) {
        let mut g = cluster.write().unwrap();
        if let Some(v) = g.get_mut(node_id) {
            if v.alive {
                tracing::warn!("node {node_id} dead (connection refused)");
            }
            v.alive = false;
            v.misses = MAX_MISSES;
        }
    }

    /// Адреси для опитування: seed-и + все, що приніс mDNS, + відомі addr із cluster.
    pub fn candidates(&self) -> Vec<(Option<String>, Vec<String>)> {
        let mut out: Vec<(Option<String>, Vec<String>)> =
            self.seeds.iter().map(|s| (None, vec![s.clone()])).collect();
        for (id, addrs) in self.resolved.lock().unwrap().iter() {
            // порожній набір (лише IPv6) не додаємо, інакше він затінив би відомий addr із cluster
            if id != &self.self_id && !addrs.is_empty() {
                out.push((Some(id.clone()), addrs.iter().cloned().collect()));
            }
        }
        for (id, v) in self.cluster.read().unwrap().iter() {
            if !out.iter().any(|(i, _)| i.as_deref() == Some(id)) {
                out.push((Some(id.clone()), vec![v.state.addr.clone()]));
            }
        }
        out
    }

    /// Анонс у mDNS + browse; помилка mDNS не фатальна — лишаються seed-peers.
    /// Демон повертається назовні: поки його тримають, анонс живий.
    fn announce_and_browse(&self) -> Option<ServiceDaemon> {
        let daemon = match ServiceDaemon::new() {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("mDNS unavailable ({e}); relying on peers");
                return None;
            }
        };
        let host = format!("llmrt-{}.local.", self.self_id);
        let props = [("node_id", self.self_id.as_str())];
        match ServiceInfo::new(
            SERVICE,
            &format!("llmrt-{}", self.self_id),
            &host,
            "",
            self.self_port,
            &props[..],
        )
        .map(ServiceInfo::enable_addr_auto)
        {
            Ok(info) => {
                if let Err(e) = daemon.register(info) {
                    tracing::warn!("mdns register: {e}");
                }
            }
            Err(e) => tracing::warn!("mdns info: {e}"),
        }
        match daemon.browse(SERVICE) {
            Ok(rx) => {
                let resolved = self.resolved.clone();
                let me = self.self_id.clone();
                tokio::spawn(async move {
                    while let Ok(ev) = rx.recv_async().await {
                        let ServiceEvent::ServiceResolved(info) = ev else {
                            continue;
                        };
                        let Some(id) = info.get_property_val_str("node_id").map(String::from)
                        else {
                            continue;
                        };
                        if id == me {
                            continue;
                        }
                        // IPv6 link-local ігноруємо (§4). Loopback анонсує enable_addr_auto,
                        // але для чужого вузла 127.0.0.1 веде на нас самих — відкидаємо.
                        // Наслідок: сусіда на хості без жодної не-loopback адреси через mDNS
                        // не дістати — його треба задати явно в `peers = [...]`, бо seed-адреси
                        // цей фільтр не проходять.
                        let port = info.get_port();
                        let addrs: HashSet<String> = info
                            .get_addresses_v4()
                            .iter()
                            .filter(|a| !a.is_loopback())
                            .map(|a| format!("{a}:{port}"))
                            .collect();
                        if addrs.is_empty() {
                            continue;
                        }
                        resolved
                            .lock()
                            .unwrap()
                            .entry(id)
                            .or_default()
                            .extend(addrs);
                    }
                    tracing::warn!("mdns browse channel closed");
                });
            }
            Err(e) => tracing::warn!("mdns browse: {e}"),
        }
        Some(daemon)
    }

    /// Звести кандидатів до одного запису на вузол: `(ключ опитування, відомий node_id, адреси)`.
    ///
    /// Seed, mDNS і запис у cluster для одного вузла дають один запис із усіма адресами — їх
    /// пробують разом, і перемагає та, що відповіла. Якби вони лишалися окремими кандидатами
    /// з однаковим ключем, перший (seed зі старою адресою) щоразу ставив би `last_poll` і
    /// затіняв решту — вузол, що переїхав на іншу адресу, назавжди лишався б мертвим.
    fn merge_candidates(
        cands: Vec<(Option<String>, Vec<String>)>,
        seed_ids: &HashMap<String, String>,
    ) -> Vec<(String, Option<String>, Vec<String>)> {
        let mut out: Vec<(String, Option<String>, Vec<String>)> = Vec::new();
        let mut at: HashMap<String, usize> = HashMap::new();
        for (known_id, addrs) in cands {
            let Some(first) = addrs.first() else { continue };
            // seed, який ми вже впізнали, зливається із записом свого вузла
            let known_id = known_id.or_else(|| seed_ids.get(first).cloned());
            let key = known_id.clone().unwrap_or_else(|| first.clone());
            match at.get(&key) {
                Some(&i) => {
                    let (_, id, known) = &mut out[i];
                    if id.is_none() {
                        *id = known_id;
                    }
                    for a in addrs {
                        if !known.contains(&a) {
                            known.push(a);
                        }
                    }
                }
                None => {
                    at.insert(key.clone(), out.len());
                    out.push((key, known_id, addrs));
                }
            }
        }
        out
    }

    /// Застосувати результат однієї проби. `known_id` — кого ми очікували на цих адресах.
    /// Повертає node_id того, хто реально відповів (якщо це не ми самі).
    ///
    /// Хто відповів, той і потрапляє в cluster; якщо це не очікуваний вузол (адресу переїхав
    /// інший хост через DHCP), очікуваному зараховується пропуск — інакше він лишався б
    /// «живим» зі старими даними назавжди.
    fn apply_probe(
        &self,
        known_id: Option<&str>,
        got: Option<(String, NodeState)>,
        last_proto_warn: &mut HashMap<String, Instant>,
    ) -> Option<String> {
        let Some((addr, st)) = got else {
            if let Some(id) = known_id {
                Self::miss(&self.cluster, id);
            }
            return None;
        };
        if let Some(id) = known_id.filter(|id| *id != st.node_id) {
            Self::miss(&self.cluster, id);
        }
        // відповіли ми самі (seed показує на нас) — ігноруємо
        if st.node_id == self.self_id {
            return None;
        }
        if st.proto != PROTO
            && last_proto_warn
                .get(&st.node_id)
                .is_none_or(|t| t.elapsed() >= PROTO_WARN_EVERY)
        {
            tracing::warn!(
                "node {} proto {} != {PROTO}, incompatible",
                st.node_id,
                st.proto
            );
            last_proto_warn.insert(st.node_id.clone(), Instant::now());
        }
        // вузол із чужим proto все одно видимий; відсіює його planner::pick
        let id = st.node_id.clone();
        Self::ingest(&self.cluster, st, &addr);
        Some(id)
    }

    /// Анонс, browse і вічний цикл опитування `/state`. Не повертається.
    pub async fn run(self) {
        let http = reqwest::Client::builder()
            .timeout(PROBE_TIMEOUT)
            .build()
            .expect("http client");
        // тримаємо демона живим, поки крутиться цикл (тобто завжди)
        let _mdns = self.announce_and_browse();

        let mut last_poll: HashMap<String, Instant> = HashMap::new();
        let mut last_proto_warn: HashMap<String, Instant> = HashMap::new();
        // seed-адреса → node_id, який на ній відповів: інакше seed лишався б «невідомим»
        // вузлом і опитувався б кожні 3 с навіть після того, як став dead.
        let mut seed_ids: HashMap<String, String> = HashMap::new();

        loop {
            // 1. звести кандидатів по вузлах і взяти тих, кому настав час
            // (блокування тут короткі, awaits немає)
            let mut due: Vec<(Option<String>, Vec<String>)> = Vec::new();
            for (key, known_id, addrs) in Self::merge_candidates(self.candidates(), &seed_ids) {
                let alive = known_id
                    .as_ref()
                    .and_then(|id| self.cluster.read().unwrap().get(id).map(|v| v.alive))
                    .unwrap_or(true);
                if last_poll
                    .get(&key)
                    .is_some_and(|t| t.elapsed() < poll_interval(alive))
                {
                    continue;
                }
                last_poll.insert(key, Instant::now());
                due.push((known_id, addrs));
            }

            // 2. проби — паралельно: інакше кожен недоступний вузол з'їдав би свої 2 с
            // з 3-секундного такту (§3)
            let probes = due.iter().map(|(_, addrs)| probe_addrs(&http, addrs));
            let results = futures_util::future::join_all(probes).await;

            // 3. застосувати результати — послідовно, під короткими блокуваннями
            for ((known_id, _), got) in due.into_iter().zip(results) {
                let answered_at = got.as_ref().map(|(a, _)| a.clone());
                let answered = self.apply_probe(known_id.as_deref(), got, &mut last_proto_warn);
                // якщо відповіла саме seed-адреса — перепривʼязуємо її до того, хто відповів:
                // адресу міг перебрати інший вузол, і тоді стара привʼязка бреше
                if let (Some(addr), Some(id)) = (answered_at, answered) {
                    if self.seeds.contains(&addr) {
                        seed_ids.insert(addr, id);
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }
}

/// Паралельно GET `/state` на всі адреси; перемагає та, що **відповіла першою** з валідним
/// `NodeState` (§4 addr). Решта проб скасовуються, тож швидка адреса не чекає на таймаут повільної.
pub async fn probe_addrs(http: &reqwest::Client, addrs: &[String]) -> Option<(String, NodeState)> {
    let mut futs: FuturesUnordered<_> = addrs
        .iter()
        .map(|a| {
            let http = http.clone();
            let a = a.clone();
            async move {
                let r = http.get(format!("http://{a}/state")).send().await.ok()?;
                let st: NodeState = r.json().await.ok()?;
                Some((a, st))
            }
        })
        .collect();
    while let Some(r) = futs.next().await {
        if r.is_some() {
            return r; // решта futures дропаються разом із FuturesUnordered
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::*;

    fn st(id: &str) -> NodeState {
        NodeState {
            proto: PROTO,
            node_id: id.into(),
            name: id.into(),
            addr: "x".into(),
            version: "0".into(),
            hw: Hw {
                cpu: "".into(),
                device: "CPU".into(),
                mem_limit_mb: 1,
            },
            models: vec![],
            free_mb: 0,
            seen: 0,
        }
    }

    #[test]
    fn ingest_updates_by_node_id_not_addr() {
        let c: SharedCluster = Default::default();
        Discovery::ingest(&c, st("a"), "10.0.0.1:7411");
        Discovery::ingest(&c, st("a"), "10.0.0.2:7411");
        let g = c.read().unwrap();
        assert_eq!(g.len(), 1);
        assert_eq!(g["a"].state.addr, "10.0.0.2:7411");
        assert!(g["a"].alive);
        assert_eq!(g["a"].misses, 0);
    }

    #[test]
    fn three_misses_then_dead_then_revive() {
        let c: SharedCluster = Default::default();
        Discovery::ingest(&c, st("a"), "10.0.0.1:7411");
        for _ in 0..2 {
            Discovery::miss(&c, "a");
            assert!(c.read().unwrap()["a"].alive);
        }
        Discovery::miss(&c, "a");
        assert!(!c.read().unwrap()["a"].alive);
        Discovery::ingest(&c, st("a"), "10.0.0.1:7411");
        assert!(c.read().unwrap()["a"].alive);
    }

    #[test]
    fn mark_dead_is_immediate() {
        let c: SharedCluster = Default::default();
        Discovery::ingest(&c, st("a"), "10.0.0.1:7411");
        Discovery::mark_dead(&c, "a");
        assert!(!c.read().unwrap()["a"].alive);
    }

    /// DHCP віддав адресу `b` іншому хосту: відповідь від `c` не робить `b` живим.
    #[test]
    fn answer_from_another_node_misses_the_expected_one() {
        let d = Discovery::new("me".into(), 7411, vec![]);
        Discovery::ingest(&d.cluster, st("b"), "10.0.0.1:7411");
        let mut warn = HashMap::new();
        let got = Some(("10.0.0.1:7411".to_string(), st("c")));
        assert_eq!(
            d.apply_probe(Some("b"), got, &mut warn).as_deref(),
            Some("c")
        );
        let g = d.cluster.read().unwrap();
        assert_eq!(g["b"].misses, 1, "той, кого чекали, пропустив опитування");
        assert_eq!(
            g["c"].state.addr, "10.0.0.1:7411",
            "хто відповів, той і в cluster"
        );
    }

    #[test]
    fn failed_probe_misses_and_own_state_is_not_ingested() {
        let d = Discovery::new("me".into(), 7411, vec![]);
        Discovery::ingest(&d.cluster, st("b"), "10.0.0.1:7411");
        let mut warn = HashMap::new();
        assert_eq!(d.apply_probe(Some("b"), None, &mut warn), None);
        assert_eq!(d.cluster.read().unwrap()["b"].misses, 1);
        // seed показує на нас самих — у cluster ми не потрапляємо
        let mine = Some(("10.0.0.9:7411".to_string(), st("me")));
        assert_eq!(d.apply_probe(None, mine, &mut warn), None);
        assert!(!d.cluster.read().unwrap().contains_key("me"));
    }

    /// Seed і mDNS для одного вузла — один кандидат з обома адресами, інакше seed зі
    /// старою адресою затіняв би нову (вузол переїхав через DHCP і переанонсувався).
    #[test]
    fn seed_and_mdns_for_one_node_merge_into_one_probe() {
        let seed_ids = HashMap::from([("10.0.0.5:7411".to_string(), "c".to_string())]);
        let cands = vec![
            (None, vec!["10.0.0.5:7411".to_string()]), // seed, уже впізнаний як c
            (Some("c".to_string()), vec!["10.0.0.9:7411".to_string()]), // mDNS: нова адреса
            (Some("d".to_string()), vec!["10.0.0.7:7411".to_string()]),
            (None, vec![]), // порожній — пропускаємо, не панікуємо
        ];
        let m = Discovery::merge_candidates(cands, &seed_ids);
        assert_eq!(m.len(), 2, "c зливається в один запис, d окремо");
        assert_eq!(m[0].0, "c");
        assert_eq!(m[0].1.as_deref(), Some("c"));
        assert_eq!(
            m[0].2,
            ["10.0.0.5:7411", "10.0.0.9:7411"],
            "обидві адреси пробуються"
        );
        assert_eq!(m[1].2, ["10.0.0.7:7411"]);
    }

    /// Невпізнаний seed лишається окремим кандидатом із ключем-адресою.
    #[test]
    fn unknown_seed_keeps_its_address_as_key() {
        let m = Discovery::merge_candidates(
            vec![(None, vec!["10.0.0.5:7411".to_string()])],
            &HashMap::new(),
        );
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].0, "10.0.0.5:7411");
        assert!(m[0].1.is_none());
    }

    #[test]
    fn poll_interval_depends_on_liveness() {
        assert_eq!(poll_interval(true), std::time::Duration::from_secs(3));
        assert_eq!(poll_interval(false), std::time::Duration::from_secs(10));
    }
}
