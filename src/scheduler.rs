use crate::component::*;
use crate::storage::*;

#[derive(Debug, Clone, Copy)]
pub enum ExecutionFrequency {
    Always,
    Once,
    //make sure smaller than tick duration
    Timed(f64, f64), //in sec
}

trait SystemParam {}
impl SystemParam for Command {}

struct Command {}
impl Command {
    pub(crate) fn new() -> Self {
        todo!()
    }

    pub fn add_component<C: Component>(comp: C) {}
}

trait SystemOutput {}
trait System {
    type Input: SystemParam;
    type Output: SystemOutput;
    fn run(&self, input: Self::Input) -> Self::Output;
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
