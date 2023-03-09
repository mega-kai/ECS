use std::any::Any;

//the scheduler and the executor
use crate::component::*;
use crate::storage::*;
use crate::system::*;
/// storing and running all the systems, generating ordered queue for those
/// systems to run, all the while requesting
pub struct Scheduler {
    //toggle this flag on when ECS::add_system invoked
    //toggle this flag off when a new queue is generated once
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
