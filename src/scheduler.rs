pub struct Scheduler {
    pool: Vec<fn()>,
}
impl Scheduler {
    pub fn new() -> Self {
        Self { pool: vec![] }
    }

    pub fn add_system(&mut self, func: fn()) {
        self.pool.push(func);
    }

    pub fn run_all(&self) {
        for system in &self.pool {
            (system)()
        }
    }
}
