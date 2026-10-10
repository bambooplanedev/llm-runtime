//! TCP-таймаути для з'єднань, які приймає gateway (B1 §2), і ті самі значення для клієнта
//! reqwest (spec A 1.1): peer, що зник без RST, виявляється за ~25 s в обидва боки.

use std::time::Duration;

/// Keepalive: перша проба після 10 s тиші, далі кожні 5 s, 3 без відповіді — обрив.
pub const KEEPALIVE_IDLE: Duration = Duration::from_secs(10);
pub const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(5);
pub const KEEPALIVE_RETRIES: u32 = 3;
/// Скільки чекати ACK на вже надіслані дані: при непідтверджених даних keepalive не діє.
pub const UNACKED_TIMEOUT: Duration = Duration::from_secs(25);

/// Без цього вузол, що шле стрім зниклому peer'у, дізнається про це через ~15 хв ретрансмітів
/// (або коли мережа повернеться) — і весь цей час тримає слот llama-server. Linux:
/// `TCP_USER_TIMEOUT`; macOS: `TCP_RXT_CONNDROPTIME` (відлік від першого ретрансміту, обрив на
/// таймері ретрансміту — фактично ~25–45 s). На Linux ≥ 5.11 `TCP_USER_TIMEOUT` рве й живого
/// клієнта, що перестав читати (нульове вікно) на > 25 s.
pub fn tune_accepted(s: &tokio::net::TcpStream) -> std::io::Result<()> {
    let sock = socket2::SockRef::from(s);
    let ka = socket2::TcpKeepalive::new()
        .with_time(KEEPALIVE_IDLE)
        .with_interval(KEEPALIVE_INTERVAL)
        .with_retries(KEEPALIVE_RETRIES);
    sock.set_tcp_keepalive(&ka)?;
    #[cfg(target_os = "linux")]
    sock.set_tcp_user_timeout(Some(UNACKED_TIMEOUT))?;
    #[cfg(target_os = "macos")]
    macos::set_rxt_conndroptime(s, UNACKED_TIMEOUT.as_secs() as libc::c_int)?;
    Ok(())
}

#[cfg(target_os = "macos")]
pub mod macos {
    use std::os::fd::AsRawFd;

    /// `<netinet/tcp.h>`: `#define TCP_RXT_CONNDROPTIME 0x80` — секунди від першого ретрансміту
    /// до обриву з'єднання. У crate `libc` 0.2.189 цієї константи нема.
    const TCP_RXT_CONNDROPTIME: libc::c_int = 0x80;

    pub fn set_rxt_conndroptime(s: &impl AsRawFd, secs: libc::c_int) -> std::io::Result<()> {
        // SAFETY: fd живий (позичений `s`), вказівник і довжина — на один `c_int`.
        let r = unsafe {
            libc::setsockopt(
                s.as_raw_fd(),
                libc::IPPROTO_TCP,
                TCP_RXT_CONNDROPTIME,
                (&secs as *const libc::c_int).cast(),
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            )
        };
        if r == 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error())
        }
    }

    pub fn rxt_conndroptime(s: &impl AsRawFd) -> std::io::Result<libc::c_int> {
        let mut v: libc::c_int = 0;
        let mut len = std::mem::size_of::<libc::c_int>() as libc::socklen_t;
        // SAFETY: як вище; `len` на вході — розмір буфера.
        let r = unsafe {
            libc::getsockopt(
                s.as_raw_fd(),
                libc::IPPROTO_TCP,
                TCP_RXT_CONNDROPTIME,
                (&mut v as *mut libc::c_int).cast(),
                &mut len,
            )
        };
        if r == 0 {
            Ok(v)
        } else {
            Err(std::io::Error::last_os_error())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn accepted_socket_gets_keepalive_and_unacked_timeout() {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let _c = tokio::net::TcpStream::connect(l.local_addr().unwrap())
            .await
            .unwrap();
        let (s, _) = l.accept().await.unwrap();
        tune_accepted(&s).unwrap();
        let sock = socket2::SockRef::from(&s);
        assert!(sock.keepalive().unwrap());
        assert_eq!(sock.tcp_keepalive_time().unwrap(), KEEPALIVE_IDLE);
        assert_eq!(sock.tcp_keepalive_interval().unwrap(), KEEPALIVE_INTERVAL);
        assert_eq!(sock.tcp_keepalive_retries().unwrap(), KEEPALIVE_RETRIES);
        #[cfg(target_os = "linux")]
        assert_eq!(sock.tcp_user_timeout().unwrap(), Some(UNACKED_TIMEOUT));
        #[cfg(target_os = "macos")]
        assert_eq!(macos::rxt_conndroptime(&s).unwrap(), 25);
    }
}
