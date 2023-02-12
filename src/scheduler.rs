use std::any::Any;

//the scheduler and the executor
use crate::command_buffer::*;
use crate::component::*;
use crate::query::*;
use crate::storage::*;
use crate::system::*;
/// storing and running all the systems, generating ordered queue for those
/// systems to run, all the while requesting
pub struct Scheduler {
    //toggle this flag on when ECS::add_system invoked
    //toggle this flag off when a new queue is generated once
    updated: bool,
    pool: Vec<Box<dyn System>>,
    queue: Vec<Box<dyn System>>,
}
impl Scheduler {
    pub fn new() -> Self {
        Self {
            pool: vec![],
            queue: vec![],
            updated: false,
        }
    }

    pub fn add_system<Param: SysParam + 'static, Sys: SysFn<Param> + 'static>(
        &mut self,
        func: SystemWithMetadata<Param, Sys>,
    ) {
        self.pool.push(Box::new(func));
        self.updated = true;
    }

    /// organzie self.queue so that it reflects on their metadata
    /// this needs to be done every tick before execution cycle
    pub fn generate_queue_for_current_tick(&mut self) {
        //1. take out all once systems from the current queue;
        //2. take out all the timed systems from the queue;
        //3. inserting all the timed systems with timers up into the queue;
        //4.

        //check the flag, if on, shove the newly added system to the queue
    }

    /// when running a system, the only state changes are components data
    /// with mutable access requested, all the commands will be executed
    /// in batch after execution phase
    pub fn execute_all(&mut self) {
        for system in &mut self.pool {
            system.run(());
        }
    }
}
