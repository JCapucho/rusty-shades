use crate::src::Span;
use std::{
    cmp::{Eq, PartialEq},
    fmt,
    ops::{Deref, DerefMut},
};

use serde::Serialize;

#[derive(Clone, Hash, Serialize)]
pub struct Node<T: Serialize, U: Serialize = Span>(Box<T>, U);

impl<T: Serialize, U: Serialize> Node<T, U> {
    pub fn new(item: T, attr: U) -> Self {
        Self(Box::new(item), attr)
    }

    pub fn inner(&self) -> &T {
        &self.0
    }

    pub fn into_inner(self) -> T {
        *self.0
    }

    pub fn map_inner<V: Serialize>(self, f: impl FnOnce(T) -> V) -> Node<V, U> {
        let Node(inner, meta) = self;
        Node::new(f(*inner), meta)
    }

    pub fn map_meta<V: Serialize>(self, f: impl FnOnce(U) -> V) -> Node<T, V> {
        let Node(inner, meta) = self;
        Node(inner, f(meta))
    }

    pub fn as_ref(&self) -> Node<&T, U>
    where
        U: Clone,
    {
        Node::new(&*self, self.1.clone())
    }

    pub fn attr(&self) -> &U {
        &self.1
    }

    pub fn attr_mut(&mut self) -> &mut U {
        &mut self.1
    }
}

impl<T: Serialize, U: Serialize> Deref for Node<T, U> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl<T: Serialize, U: Serialize> DerefMut for Node<T, U> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.0
    }
}

impl<T: fmt::Debug + Serialize, U: fmt::Debug + Serialize> fmt::Debug for Node<T, U> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if f.alternate() {
            write!(f, "({:#?}: {:#?})", self.0, self.attr())
        } else {
            write!(f, "({:?}: {:?})", self.0, self.attr())
        }
    }
}

impl<T: PartialEq + Serialize, U: Serialize> PartialEq for Node<T, U> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: Eq + Serialize, U: Serialize> Eq for Node<T, U> {}

pub type SrcNode<T> = Node<T, Span>;

impl<T: Serialize> SrcNode<T> {
    pub fn span(&self) -> Span {
        *self.attr()
    }
}
