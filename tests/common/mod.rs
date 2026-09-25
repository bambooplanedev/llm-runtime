//! Спільне для тестових бінарників, що займають фіксовані порти.

use std::fs::File;
use std::os::fd::AsRawFd;
use std::sync::OnceLock;

/// Порти дітей і демонів у тестах фіксовані й спільні для всієї машини. Два тестові процеси
/// одночасно (інший checkout, worktree, сесія) б'ються за них: bind-проба `free_port` і bind
/// дитини — не атомарні між процесами, дитина падає з `AddrInUse`, модель іде в `Failed`.
/// Міжпроцесний lock тримається до кінця процесу: другий `cargo test` на машині чекає.
pub fn machine_port_lock() {
    static LOCK: OnceLock<File> = OnceLock::new();
    LOCK.get_or_init(|| {
        let path = std::env::temp_dir().join("llmrt-test-ports.lock");
        let f = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
        let fd = f.as_raw_fd();
        // SAFETY: `fd` живий, поки живий `f`, а `f` лежить у static до кінця процесу.
        if unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            eprintln!(
                "waiting for another llmrt test process (lock {})",
                path.display()
            );
            // SAFETY: як вище.
            let r = unsafe { libc::flock(fd, libc::LOCK_EX) };
            assert_eq!(
                r,
                0,
                "flock {}: {}",
                path.display(),
                std::io::Error::last_os_error()
            );
        }
        f
    });
}
