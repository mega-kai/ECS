use crate::component::*;
use crate::storage::*;

#[derive(Debug, Clone, Copy)]
pub enum ExecutionFrequency {
    Always,
    Once,
    //make sure smaller than tick duration
    Timed(f64, f64), //in sec
}

pub struct Scheduler {
    updated: bool,
    pool: (),
    queue: (),
}
impl Scheduler {
    pub fn new() -> Self {
        Self {
            pool: (),
            queue: (),
            updated: false,
        }
    }

    pub fn add_system(&mut self) {
        self.updated = true;
    }

    pub fn generate_queue_for_current_tick(&mut self) {}

    pub fn execute_all(&mut self) {}
}
