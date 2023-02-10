use crate::component::*;

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
    query_request: QueryRequest,
    fn_ptr: fn(CommandQueue, QueryResult),
}
impl System {
    /// generate a stub system that runs every game tick
    pub fn default(fn_ptr: fn(CommandQueue, QueryResult)) -> Self {
        Self {
            fn_ptr,
            query_request: QueryRequest::empty(),
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

    pub fn run_all(&self) {
        for system in &self.queue {
            (system.fn_ptr)(CommandQueue::new(), QueryResult::new());
        }
    }
}
