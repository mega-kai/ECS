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

    #[test]
    fn storage() {
        let mut storage = ComponentTable::new();

        // add and retrieve
        let name0 = "pl 0";
        let player0 = Player(name0);
        let key0 = storage.insert(player0);
        let ref0 = storage.get_as::<Player>(key0).unwrap();
        assert_eq!(ref0.0, name0);

        // remove
        let remove0 = storage.remove_as::<Player>(key0).unwrap();
        assert_eq!(remove0.0, name0);

        // error: mismatched types
        let name1 = "pl 1";
        let player1 = Player(name1);
        let key1 = storage.insert(player1);
        let err1 = storage.get_as::<Mana>(key1);
        assert_eq!(err1.unwrap_err(), "generic and the key don't match");

        // error: wrong key index
        let name2 = "pl 2";
        let player2 = Player(name2);
        let key2 = storage.insert(player2);
        let err2 = storage
            .get_as::<Player>(ComponentAccess::new_from_type::<Player>(999))
            .unwrap_err();
        // println!("{}", err2);
        assert_eq!(err2, "index overflow in dense vec");

        // error: wrong key comp id
        let name3 = "pl 3";
        let player3 = Player(name3);
        let key3 = storage.insert(player3);
        let err3 = storage
            .get_as::<Mana>(ComponentAccess::new_from_type::<Mana>(999))
            .unwrap_err();
        // println!("{}", err3);
        assert_eq!(err3, "no such component type exist in this storage");
    }
}
