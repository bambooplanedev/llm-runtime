//! Спільне для тестових бінарників, що займають фіксовані порти.

use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::sync::OnceLock;

/// Шлях фіксований, а не `std::env::temp_dir()`: порти спільні для всієї машини, а `TMPDIR`
/// різниться між оболонками й сесіями (nix shell, пісочниця) — lock у ньому не змусив би
/// чекати другий такий прогін на тій самій машині.
const LOCK_PATH: &str = "/tmp/llmrt-test-ports.lock";

/// Порти дітей і демонів у тестах фіксовані й спільні для всієї машини. Два тестові процеси
/// одночасно (інший checkout, worktree, сесія) б'ються за них: bind-проба `free_port` і bind
/// дитини — не атомарні між процесами, дитина падає з `AddrInUse`, модель іде в `Failed`.
/// Міжпроцесний lock тримається до кінця процесу: другий `cargo test` на машині чекає.
/// Кожен новий тестовий бінарник, що займає фіксовані порти, повинен викликати
/// `machine_port_lock()` першим (як `cfg()` у tests/runner.rs і `serial()` у tests/integration.rs).
pub fn machine_port_lock() {
    static LOCK: OnceLock<File> = OnceLock::new();
    LOCK.get_or_init(|| {
        let path = LOCK_PATH;
        let f = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)
            .or_else(|e| {
                if e.kind() == std::io::ErrorKind::PermissionDenied {
                    // Файл створив інший користувач: запис недоступний, але flock працює і на
                    // read-only fd (Linux, macOS).
                    std::fs::OpenOptions::new().read(true).open(path)
                } else {
                    Err(e)
                }
            })
            .unwrap_or_else(|e| panic!("open {path}: {e}"));
        let fd = f.as_raw_fd();
        // SAFETY: `fd` живий, поки живий `f`, а `f` лежить у static до кінця процесу.
        if unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            // Пряме письмо у stderr, а не `eprintln!`: libtest перехоплює print!/eprint! і
            // ховає їх для тесту, що зрештою пройшов, — тоді очікування виглядало б як завислий
            // процес. Прямий запис у `io::stderr()` обходить цей перехоплювач.
            let _ = writeln!(
                std::io::stderr(),
                "waiting for another llmrt test process (lock {path})"
            );
            loop {
                // SAFETY: як вище.
                let r = unsafe { libc::flock(fd, libc::LOCK_EX) };
                if r == 0 {
                    break;
                }
                let err = std::io::Error::last_os_error();
                // EINTR — сигнал перервав очікування (напр. libtest-таймер): пробуємо ще раз.
                if err.raw_os_error() != Some(libc::EINTR) {
                    panic!("flock {path}: {err}");
                }
            }
        }
        f
    });
}
