#![allow(dead_code, unused_variables, unused_imports, unused_mut)]
#![feature(alloc_layout_extra, map_try_insert)]
mod component;
mod scheduler;
mod storage;
use component::*;
use scheduler::*;
use storage::*;

pub struct ECS {
    storage: ComponentTable,
    scheduler: Scheduler,
}

impl ECS {
    pub fn new() -> Self {
        Self {
            storage: ComponentTable::new(),
            scheduler: Scheduler::new(),
        }
    }

    pub fn add_system(&mut self, system: System) {
        self.scheduler.add_system(system);
    }

    pub fn tick(&mut self) {
        self.scheduler.prepare_queue();
        for system in &self.scheduler.queue {
            system.run(&mut self.storage);
        }
    }
}

#[cfg(test)]
mod test {
    use std::alloc::Layout;

    use super::*;

    #[derive(Clone)]
    struct Health(i32);
    impl Component for Health {}

    #[derive(Clone, Debug)]
    struct Mana(i32);
    impl Component for Mana {}

    #[derive(Clone, Debug)]
    struct Player(&'static str);
    impl Component for Player {}

    fn system(mut command: Command) {
        command.add_component(Player("uwu"));
        for val in command.query::<Player, ()>() {
            println!("{}", val.0);
        }
    }
}
