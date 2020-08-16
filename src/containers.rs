//All my Vector-backed data structures use !0 for NULL
macro_rules! NULL {
    () => {
        !0
    };
}

use std::ops;
use std::iter;

static RIGHT: usize = 1;
static LEFT: usize = 0;

pub struct Node2<T> {
    pub children: [u32; 2],
    pub data: Option<T>,
}

impl<T> Node2<T> {
    pub fn left_mut(&mut self) -> &mut u32 {
        &mut self.children[0]
    }
    pub fn right_mut(&mut self) -> &mut u32 {
        &mut self.children[1]
    }
    pub fn left_ref(&mut self) -> &u32 {
        &self.children[0]
    }
    pub fn right_ref(&mut self) -> &u32 {
        &self.children[1]
    }
}

pub struct DoublyLinkedList<T> {
    memory: Vec<Node2<T>>,
    front: u32,
    rear: u32,
    node_pool: u32,
    size: u32,
}

impl<T> DoublyLinkedList<T>
where
    Self: ops::Index<u32, Output = Node2<T>> + ops::IndexMut<u32, Output = Node2<T>>,
{
    pub fn new() -> DoublyLinkedList<T> {
        let mut dll = DoublyLinkedList {
            memory: Vec::new(),
            front: NULL!(),
            rear: NULL!(),
            node_pool: NULL!(),
            size: 0,
        };
        dll.allocate_node(None);
        dll.front = 0;
        dll.rear = 0;
        dll
    }

    pub fn push_front(&mut self, item: T) {
        let node_ptr = self.allocate_node(Some(item));
        let mask = zero_mask(self.size);
        let old_front = self.front;
        *self[old_front].left_mut() = node_ptr;
        *self[node_ptr].right_mut() = old_front;
        *self[node_ptr].left_mut() = self.rear;
        self.front = node_ptr;
        //optionally update rear pointer
        self.rear = node_ptr & (!mask) | self.rear & mask;
        self.size += 1;
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() == false {
            let old_front = self.front;
            let new_front = self[old_front].children[RIGHT];
            self.front = new_front;
            self[new_front].children[LEFT] = self.rear;
            let removed_item = self[old_front].data.take();
            self.delete_node(old_front);
            self.size-=1;
            removed_item
        } else {
            None
        }
    }

    pub fn push_rear(&mut self, item: T) {
        let node_ptr = self.allocate_node(Some(item));
        let mask = zero_mask(self.size);
        let old_rear = self.rear;
        *self[old_rear].right_mut() = node_ptr;
        *self[node_ptr].left_mut() = old_rear;
        *self[node_ptr].right_mut() = self.front;
        self.rear = node_ptr;
        //optionally update front  pointer
        self.front = node_ptr & (!mask) | self.front & mask;
        self.size += 1;
    }

    pub fn pop_rear(&mut self) -> Option<T> {
        if self.is_empty() == false {
            let old_rear = self.rear;
            let new_rear = self[old_rear].children[LEFT];
            
            self.rear = new_rear;
            self[new_rear].children[RIGHT] = self.front;
            
            let removed_item = self[old_rear].data.take();
            self.delete_node(old_rear);
            self.size-=1;
            removed_item
        } else {
            None
        }
    }


    pub fn node_iter(&self)->DLLNodeIterator<T>{
        DLLNodeIterator {
            front_ptr: self.front,
            rear_ptr: self.rear, 
            size: self.size,
            dll_ref: self,
        }
    }

    pub fn iter(&self) -> DLLIterator<T> {
        DLLIterator {
            front_ptr: self.front,
            size: self.size,
            dll_ref: self,
        }
    }

    pub fn allocate_node(&mut self, item: Option<T>) -> u32 {
        if self.node_pool != NULL!() {
            let node_ptr = self.node_pool;
            //update node pointer
            self.node_pool = *self[node_ptr].right_ref();
            //clear node of previous data
            *self[node_ptr].left_mut() = NULL!();
            *self[node_ptr].right_mut() = NULL!();
            self[node_ptr].data = item;
            node_ptr
        } else {
            //push a new node into memory
            self.memory.push(Node2 {
                data: item,
                children: [NULL!(); 2],
            });
            //return the nodes address
            (self.memory.len() - 1) as u32
        }
    }
    ///removes node from the list and adds it to the node pool;
    fn delete_node(&mut self, ptr: u32) {
        if self.node_pool == NULL!() {
            self.node_pool = ptr;
            self[ptr].children = [NULL!(); 2];
        } else {
            self[ptr].children[RIGHT] = self.node_pool;
            self[ptr].children[LEFT] = NULL!();
            self.node_pool = ptr;
        }
    }
}

impl<T> ops::Index<u32> for DoublyLinkedList<T> {
    type Output = Node2<T>;
    fn index(&self, index: u32) -> &Self::Output {
        &self.memory[index as usize]
    }
}

impl<T> ops::IndexMut<u32> for DoublyLinkedList<T> {
    fn index_mut(&mut self, index: u32) -> &mut Self::Output {
        &mut self.memory[index as usize]
    }
}

pub struct DLLIterator<'a, T> {
    pub front_ptr: u32,
    pub size: u32,
    pub dll_ref: &'a DoublyLinkedList<T>,
}

impl<'a, T> std::iter::Iterator for DLLIterator<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.size > 0 {
            self.size -= 1;
            let old_front = self.front_ptr;
            self.front_ptr = self.dll_ref[old_front].children[RIGHT];
            Some(&self.dll_ref[old_front].data.as_ref().unwrap())
        } else {
            None
        }
    }
}

pub struct DLLNodeIterator<'a, T> {
    pub front_ptr: u32,
    pub rear_ptr: u32,
    pub size: u32,
    pub dll_ref: &'a DoublyLinkedList<T>,
}

impl<'a, T> std::iter::Iterator for DLLNodeIterator<'a, T> {
    type Item = &'a Node2<T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.size > 0 {
            self.size -= 1;
            let old_front = self.front_ptr;
            self.front_ptr = self.dll_ref[old_front].children[RIGHT];
            Some(&self.dll_ref[old_front])
        } else {
            None
        }
    }
}


impl<'a, T> iter::DoubleEndedIterator for DLLNodeIterator<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.size > 0 {
            self.size -= 1;
            let old_rear = self.rear_ptr;
            self.rear_ptr = self.dll_ref[old_rear].children[LEFT];
            Some(&self.dll_ref[old_rear])
        } else {
            None
        }
    }
}


/// If 'num' is not 0 it returns the value : 0xff..ff. Otherwise it returns 0
pub fn zero_mask(num: u32) -> u32 {
    ((  (-(num as i32)) as u32 | num) as i32 >> 31) as u32
}




#[test]
pub fn dll_test1(){
    let mut dll = DoublyLinkedList::<i32>::new();
    dll.push_rear(1);
    dll.push_rear(2);
    dll.push_front(-1);
    let result = dll.iter().map(|x| *x).collect::<Vec<_>>(); 
    assert_eq!( result , vec![-1,1,2]);
}