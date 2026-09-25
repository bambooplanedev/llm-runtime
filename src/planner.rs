use crate::config::Tiers;
use crate::state::{Cluster, ModelState, PROTO};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Small,
    Medium,
    Large,
}

#[derive(Debug, Clone)]
pub enum Want {
    Tier(Tier),
    Model(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pair {
    pub node_id: String,
    pub model_id: String,
}

#[derive(Debug, Clone)]
pub struct Pick {
    pub pair: Pair,
    pub addr: String,
    pub cold: bool,
    /// Модель уже вантажилась на момент вибору: ми приєдналися до чужого старту.
    pub joined: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoPick {
    UnknownModel,
    NoSuchTier,
    AllBusy,
}

pub const TIER_NAMES: [&str; 4] = ["small", "medium", "large", "auto"];

pub fn parse_want(s: &str) -> Option<Want> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    Some(match s.to_ascii_lowercase().as_str() {
        "small" | "auto" => Want::Tier(Tier::Small),
        "medium" => Want::Tier(Tier::Medium),
        "large" => Want::Tier(Tier::Large),
        _ => Want::Model(s.to_ascii_lowercase()),
    })
}

pub fn tier_of(params_b: f64, t: &Tiers) -> Tier {
    if params_b < t.small {
        Tier::Small
    } else if params_b <= t.medium {
        Tier::Medium
    } else {
        Tier::Large
    }
}

pub fn pick(
    cluster: &Cluster,
    want: &Want,
    tiers: &Tiers,
    exclude: &HashSet<Pair>,
) -> Result<Pick, NoPick> {
    let mut any_match = false;
    let mut best: Option<(Pick, (u8, f64, i64))> = None;
    for (node_id, view) in cluster {
        if !view.alive || view.state.proto != PROTO {
            continue;
        }
        for m in &view.state.models {
            let matches = match want {
                Want::Tier(t) => tier_of(m.params_b, tiers) == *t,
                Want::Model(id) => &m.id == id,
            };
            if !matches {
                continue;
            }
            any_match = true;
            let pair = Pair {
                node_id: node_id.clone(),
                model_id: m.id.clone(),
            };
            if exclude.contains(&pair) {
                continue;
            }
            // Ранг стану: Loaded, потім Loading (чужий холодний старт уже йде, його пам'ять
            // вузол уже врахував — free_mb не перевіряємо), потім Available, якщо влазить.
            let rank = match m.state {
                ModelState::Loaded => 0u8,
                ModelState::Loading => 1,
                ModelState::Available if view.state.free_mb >= m.need_mb => 2,
                _ => continue,
            };
            // Менший ключ виграє: (ранг стану, -params_b, inflight - slots)
            let key = (rank, -m.params_b, m.inflight as i64 - m.slots as i64);
            let cand = Pick {
                pair,
                addr: view.state.addr.clone(),
                cold: rank > 0,
                joined: rank == 1,
            };
            let better = match &best {
                None => true,
                Some((_, k)) => key.partial_cmp(k) == Some(std::cmp::Ordering::Less),
            };
            if better {
                best = Some((cand, key));
            }
        }
    }
    match best {
        Some((p, _)) => Ok(p),
        None if !any_match => Err(match want {
            Want::Model(_) => NoPick::UnknownModel,
            Want::Tier(_) => NoPick::NoSuchTier,
        }),
        None => Err(NoPick::AllBusy),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::*;
    use std::collections::{HashMap, HashSet};
    use std::time::Instant;

    fn model(
        id: &str,
        params_b: f64,
        st: ModelState,
        need: u64,
        inflight: u32,
        slots: u32,
    ) -> ModelEntry {
        ModelEntry {
            id: id.into(),
            file: format!("{id}.gguf"),
            params_b,
            active_params_b: None,
            need_mb: need,
            state: st,
            slots,
            inflight,
        }
    }
    fn node(id: &str, free_mb: u64, models: Vec<ModelEntry>) -> (String, NodeView) {
        (
            id.into(),
            NodeView {
                alive: true,
                last_seen: Instant::now(),
                misses: 0,
                state: NodeState {
                    proto: PROTO,
                    node_id: id.into(),
                    name: id.into(),
                    addr: format!("10.0.0.{}:7411", id.len()),
                    version: "0.1.0".into(),
                    hw: Hw {
                        cpu: "".into(),
                        device: "CPU".into(),
                        mem_limit_mb: 0,
                    },
                    models,
                    free_mb,
                    seen: 0,
                },
            },
        )
    }
    fn tiers() -> crate::config::Tiers {
        crate::config::Tiers {
            small: 3.0,
            medium: 12.0,
        }
    }
    fn small() -> Want {
        Want::Tier(Tier::Small)
    }

    #[test]
    fn parse_want_and_tiers() {
        assert!(matches!(parse_want("small"), Some(Want::Tier(Tier::Small))));
        assert!(matches!(parse_want("auto"), Some(Want::Tier(Tier::Small))));
        assert!(matches!(parse_want("LARGE"), Some(Want::Tier(Tier::Large))));
        assert!(matches!(
            parse_want("qwen3-8b-q4_k_m"),
            Some(Want::Model(_))
        ));
        assert_eq!(tier_of(2.9, &tiers()), Tier::Small);
        assert_eq!(tier_of(3.0, &tiers()), Tier::Medium);
        assert_eq!(tier_of(12.1, &tiers()), Tier::Large);
    }

    #[test]
    fn no_candidates_for_tier() {
        let c: Cluster = HashMap::from([node(
            "a",
            9000,
            vec![model("m8", 8.0, ModelState::Loaded, 6000, 0, 1)],
        )]);
        assert!(matches!(
            pick(&c, &small(), &tiers(), &HashSet::new()),
            Err(NoPick::NoSuchTier)
        ));
    }

    #[test]
    fn loaded_beats_available() {
        let c: Cluster = HashMap::from([
            node(
                "a",
                9000,
                vec![model("m2", 2.0, ModelState::Available, 1500, 0, 1)],
            ),
            node(
                "bb",
                9000,
                vec![model("m1", 1.0, ModelState::Loaded, 1000, 0, 1)],
            ),
        ]);
        let p = pick(&c, &small(), &tiers(), &HashSet::new()).unwrap();
        assert_eq!(p.pair.node_id, "bb");
        assert!(!p.cold);
    }

    #[test]
    fn params_beats_inflight_within_tier() {
        let c: Cluster = HashMap::from([
            node(
                "a",
                9000,
                vec![model("m1", 1.7, ModelState::Loaded, 1000, 0, 1)],
            ),
            node(
                "bb",
                9000,
                vec![model("m2", 2.9, ModelState::Loaded, 2000, 3, 1)],
            ),
        ]);
        assert_eq!(
            pick(&c, &small(), &tiers(), &HashSet::new())
                .unwrap()
                .pair
                .node_id,
            "bb"
        );
    }

    #[test]
    fn same_model_less_queue_wins_using_slots() {
        let c: Cluster = HashMap::from([
            node(
                "a",
                9000,
                vec![model("m1", 1.7, ModelState::Loaded, 1000, 2, 4)],
            ), // 2-4 = -2
            node(
                "bb",
                9000,
                vec![model("m1", 1.7, ModelState::Loaded, 1000, 1, 1)],
            ), // 1-1 = 0
        ]);
        assert_eq!(
            pick(&c, &small(), &tiers(), &HashSet::new())
                .unwrap()
                .pair
                .node_id,
            "a"
        );
    }

    #[test]
    fn available_needs_free_memory() {
        let c: Cluster = HashMap::from([node(
            "a",
            1000,
            vec![model("m2", 2.0, ModelState::Available, 1500, 0, 1)],
        )]);
        assert!(matches!(
            pick(&c, &small(), &tiers(), &HashSet::new()),
            Err(NoPick::AllBusy)
        ));
    }

    #[test]
    fn exclude_dead_wrong_proto_failed_draining() {
        let mut c: Cluster = HashMap::from([
            node(
                "a",
                9000,
                vec![model("m1", 1.0, ModelState::Loaded, 1000, 0, 1)],
            ),
            node(
                "bb",
                9000,
                vec![model("m1", 1.0, ModelState::Loaded, 1000, 0, 1)],
            ),
            node(
                "ccc",
                9000,
                vec![model("m1", 1.0, ModelState::Loaded, 1000, 0, 1)],
            ),
            node(
                "dddd",
                9000,
                vec![
                    model("m1", 1.0, ModelState::Failed, 1000, 0, 1),
                    model("m9", 1.0, ModelState::Draining, 1000, 0, 1),
                ],
            ),
        ]);
        c.get_mut("bb").unwrap().alive = false;
        c.get_mut("ccc").unwrap().state.proto = 99;
        let ex = HashSet::from([Pair {
            node_id: "a".into(),
            model_id: "m1".into(),
        }]);
        assert!(matches!(
            pick(&c, &small(), &tiers(), &ex),
            Err(NoPick::AllBusy)
        ));
        assert_eq!(
            pick(&c, &small(), &tiers(), &HashSet::new())
                .unwrap()
                .pair
                .node_id,
            "a"
        );
    }

    #[test]
    fn loading_ranks_between_loaded_and_available() {
        // Loading не потребує free_mb: пам'ять під неї вузол уже врахував.
        let c: Cluster = HashMap::from([
            node(
                "a",
                9000,
                vec![model("m1", 1.0, ModelState::Available, 1000, 0, 1)],
            ),
            node(
                "bb",
                0,
                vec![model("m1", 1.0, ModelState::Loading, 1000, 0, 1)],
            ),
        ]);
        let p = pick(&c, &small(), &tiers(), &HashSet::new()).unwrap();
        assert_eq!(p.pair.node_id, "bb");
        assert!(p.cold && p.joined);

        let c: Cluster = HashMap::from([
            node(
                "a",
                9000,
                vec![model("m2", 2.5, ModelState::Loading, 1000, 0, 1)],
            ),
            node(
                "bb",
                9000,
                vec![model("m1", 1.0, ModelState::Loaded, 1000, 0, 1)],
            ),
        ]);
        let p = pick(&c, &small(), &tiers(), &HashSet::new()).unwrap();
        assert_eq!(
            p.pair.node_id, "bb",
            "Loaded beats Loading even with fewer params"
        );
        assert!(!p.cold && !p.joined);
    }

    #[test]
    fn explicit_model_picks_node_only() {
        let c: Cluster = HashMap::from([
            node(
                "a",
                9000,
                vec![model("m1", 1.0, ModelState::Loaded, 1000, 0, 1)],
            ),
            node(
                "bb",
                9000,
                vec![model("m8", 8.0, ModelState::Loaded, 6000, 0, 1)],
            ),
        ]);
        let w = Want::Model("m8".into());
        assert_eq!(
            pick(&c, &w, &tiers(), &HashSet::new())
                .unwrap()
                .pair
                .node_id,
            "bb"
        );
        assert!(matches!(
            pick(&c, &Want::Model("nope".into()), &tiers(), &HashSet::new()),
            Err(NoPick::UnknownModel)
        ));
    }
}
