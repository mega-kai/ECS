#![allow(dead_code, unused_variables, unused_imports)]
#![feature(alloc_layout_extra)]
mod component;
mod scheduler;
mod storage;
use component::*;
use scheduler::*;
use storage::*;

/// the entity component system;
/// the gist of it is at any given time during any system of the same
/// execution cycle(within the same next() method) run, only one
/// mutable reference to a single component can be issued,
/// with that reference you can edit/delete that component
struct ECS {
    //for all components (including all entities)
    storage: Storage,
    //for all systems along with their metadata
    scheduler: Scheduler,
    //commands collected and to be executed and emptied each cycle ends
    command_buffer: CommandBuffer,
}

/// these are APIs to interact with the ECS, should be as simple as possible
impl ECS {
    /// not much to be said about this, just init new empty containers
    /// and all that
    pub fn new() -> Self {
        Self {
            storage: Storage::new(),
            scheduler: Scheduler::new(),
            command_buffer: CommandBuffer::new(),
        }
    }

    /// call the add_system on self.storage, essentially just pushing
    /// the system object onto a vector of systems, while toggling a flag
    /// that says a new system had been added
    pub fn add_system(&mut self, func: System) {
        self.scheduler.add_system(func);
    }

    /// a tick cycle consisting of multiple stages
    pub fn next(&mut self) {
        //generating a new queue to be executed: queue generation stages
        //this is all side effects, internally mutating the queue field
        let order = self.scheduler.generate_queue_for_current_cycle();

        //supplying with all the requested query results: query stage

        //executing all the systems in that queue sequentially: execution stage
        self.scheduler.execute_all();

        //collecting all the commands and executing them: command stage
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    struct Test(i32);
    impl Component for Test {}

    //function is a system's vital part, has a standard format of input
    //and output
    fn spawn(mut command: Command, mut result: QueryResult) {
        println!("hello ecs!");
        //spawn a bunch of components and link them together
        command.spawn_component();
        command.spawn_component();
        command.attach_component_to_another();
        //do some manipulation on the results
        result.does_things_with_the_result();
    }

    #[test]
    fn ecs() {
        //new ecs
        let mut ecs = ECS::new();

        //add systems
        ecs.add_system(System::default(spawn));

        //tick cycling
        loop {
            ecs.next();
        }
    }
}
