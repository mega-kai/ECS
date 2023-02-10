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
    storage: Storage,
    scheduler: Scheduler,
}
impl ECS {
    pub fn new() -> Self {
        Self {
            storage: Storage::new(),
            scheduler: Scheduler::new(),
        }
    }

    /// one cycle equals one tick
    pub fn next(&mut self) {
        let order = self.scheduler.generate_queue_for_current_cycle();
        self.scheduler.run_all();
    }

    pub fn add_system(&mut self, func: System) {
        self.scheduler.add_system(func);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    struct Test(i32);
    impl Component for Test {}

    fn spawn(mut command: CommandQueue, mut result: QueryResult) {
        println!("hello ecs!");
        //spawn a bunch of components and link them together
        command.spawn_component();
        command.spawn_component();
        command.attach_component_to_another();
        //do some manipulation on the results
        result.does_things_with_the_result();
    }

    #[test]
    fn sparse_set() {
        let mut ecs = ECS::new();
        ecs.add_system(System::default(spawn));
        ecs.next();

        // what is the key used for??
        //let key = ecs.spawn(Test(12));

        //let exclusive_reference_to_test = ecs.query(QueryRequest::new()).unwrap();
        // exclusive_reference_to_test.0 += 1;
    }
}
