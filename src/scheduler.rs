use crate::component::*;
use crate::storage::*;

/// a command buffer for a single system, supplied to all systems upon
/// execution, returned by that function to be collected within
/// the main ECS::CommandBuffer;
/// all commands here will only get executed at a later stage of a tick
/// cycle, which is after the execution phase
pub struct Command {}
impl Command {
    fn new() -> Self {
        todo!()
    }

    pub fn spawn_component<C: Component>(&mut self, component: C) -> ComponentKey {
        todo!()
    }

    //should you need an exclusive access to that component to destroy it??
    pub fn destroy_component<C: Component>(&mut self, mut_access: &mut C) {}

    /// perhaps the storage also stores linking information alongside with
    /// the actual component data
    pub fn link_component(&mut self, one: ComponentKey, another: ComponentKey) {}

    pub fn unlink_component(&mut self) {}
}

/// a list of desired references with filter functionalities;
/// part of the metadata, as with all other metadatas, it will be
/// processed by the scheduler
#[derive(Debug, Clone, Copy)]
pub struct QueryList {}
impl QueryList {
    pub fn empty() -> Self {
        Self {}
    }
}

/// a single query result that can be iterated upon to get all the references
/// to the components/entities
pub struct QueryResult(/* a bunch of references to components */);
impl QueryResult {
    pub fn new() -> Self {
        todo!()
    }

    pub fn does_things_with_the_result(&mut self) {}
}

#[derive(Debug, Clone, Copy)]
pub enum ExecutionFrequency {
    Always,
    Once,
    //make sure it's not smaller than the tick duration
    Timed(f64), //in sec
}

/// a system containing metadata
#[derive(Clone, Copy, Debug)]
pub struct System {
    //add ordering information
    execution_frequency: ExecutionFrequency,
    //no dynamic querying
    query_request: QueryList,
    fn_ptr: fn(Command, QueryResult) -> Command,
}
impl System {
    /// generate a stub system that runs every game tick
    pub fn default(fn_ptr: fn(Command, QueryResult) -> Command) -> Self {
        Self {
            fn_ptr,
            query_request: QueryList::empty(),
            execution_frequency: ExecutionFrequency::Always,
        }
    }
}

pub struct QueryOrder {}
impl QueryOrder {
    fn empty() -> Self {
        Self {}
    }
}

/// storing and running all the systems, generating ordered queue for those
/// systems to run, all the while requesting
pub struct Scheduler {
    //toggle this flag on when ECS::add_system invoked
    //toggle this flag off when a new queue is generated once
    updated: bool,
    pool: Vec<System>,
    queue: Vec<System>,
}
impl Scheduler {
    pub fn new() -> Self {
        Self {
            pool: vec![],
            queue: vec![],
            updated: false,
        }
    }

    pub fn add_system(&mut self, func: System) {
        //hoops the run thru before
        self.pool.push(func);
        self.updated = true;
    }

    /// organzie self.queue so that it reflects on their metadata
    /// this needs to be done every before execution cycle
    pub fn generate_queue_for_current_cycle(&mut self) -> QueryOrder {
        //first take out all once systems;
        //second take out all the timed

        //check the flag, if on, shove the newly added system to the queue

        //sort the thing
        //generate a query order from the sorted systems
        QueryOrder::empty()
    }

    pub fn execute_all(&self) {
        for system in &self.queue {
            (system.fn_ptr)(Command::new(), QueryResult::new());
        }
    }
}
