/*!
 * SYNTH Type System
 * Provides type definitions, inference, and checking
 */

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use crate::ast;

/// Core type definitions
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    /// Primitive types
    Unit,
    Bool,
    Int,
    Float,
    String,
    
    /// Compound types
    Array(Box<Type>),
    Tuple(Vec<Type>),
    Optional(Box<Type>),
    
    /// AI-native types
    Uncertain(Box<Type>),
    Embedding(usize), // dimension
    Semantic(Box<Type>),
    
    /// Function types
    Function(FunctionType),
    
    /// User-defined types
    Struct(StructType),
    Enum(EnumType),
    Interface(InterfaceType),
    
    /// Template types
    Template(TemplateType),
    
    /// Generic types
    TypeParameter(String),
    Generic(String, Vec<Type>), // name, type arguments
    
    /// Special types
    Any,
    Never,
    Unknown,
    Infer, // Type to be inferred
    
    /// Cross-domain types
    Domain(String, Box<Type>), // domain name, wrapped type
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionType {
    pub params: Vec<Type>,
    pub return_type: Box<Type>,
    pub is_async: bool,
    pub is_pure: bool,
    pub effects: Vec<Effect>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructType {
    pub name: String,
    pub fields: HashMap<String, Type>,
    pub type_params: Vec<String>,
    pub implements: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumType {
    pub name: String,
    pub variants: HashMap<String, Option<Type>>,
    pub type_params: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InterfaceType {
    pub name: String,
    pub methods: HashMap<String, FunctionType>,
    pub type_params: Vec<String>,
    pub extends: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TemplateType {
    pub name: String,
    pub props: HashMap<String, Type>,
    pub slots: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Effect {
    IO,
    Network,
    AI,
    Quantum,
    State,
    Random,
    Time,
}

impl Type {
    /// Check if type is numeric
    pub fn is_numeric(&self) -> bool {
        matches!(self, Type::Int | Type::Float)
    }
    
    /// Check if type is comparable
    pub fn is_comparable(&self) -> bool {
        matches!(
            self,
            Type::Bool | Type::Int | Type::Float | Type::String
        )
    }
    
    /// Check if type is iterable
    pub fn is_iterable(&self) -> bool {
        matches!(
            self,
            Type::Array(_) | Type::String
        )
    }
    
    /// Get the confidence type for uncertain values
    pub fn confidence_type(&self) -> Type {
        Type::Float
    }
    
    /// Check if type has AI operations
    pub fn supports_ai_ops(&self) -> bool {
        matches!(
            self,
            Type::String | Type::Embedding(_) | Type::Semantic(_) | Type::Uncertain(_)
        )
    }
    
    /// Check if types are compatible
    pub fn is_compatible_with(&self, other: &Type) -> bool {
        match (self, other) {
            (Type::Any, _) | (_, Type::Any) => true,
            (Type::Unknown, _) | (_, Type::Unknown) => true,
            (Type::Infer, _) | (_, Type::Infer) => true,
            (Type::Never, _) | (_, Type::Never) => false,
            
            // Exact matches
            (a, b) if a == b => true,
            
            // Numeric coercion
            (Type::Int, Type::Float) | (Type::Float, Type::Int) => true,
            
            // Optional compatibility
            (Type::Optional(a), Type::Optional(b)) => a.is_compatible_with(b),
            (a, Type::Optional(b)) => a.is_compatible_with(b),
            
            // Uncertain compatibility
            (Type::Uncertain(a), Type::Uncertain(b)) => a.is_compatible_with(b),
            (a, Type::Uncertain(b)) => a.is_compatible_with(b),
            
            // Array compatibility
            (Type::Array(a), Type::Array(b)) => a.is_compatible_with(b),
            
            // Domain compatibility
            (Type::Domain(_, a), Type::Domain(_, b)) => a.is_compatible_with(b),
            (Type::Domain(_, a), b) | (a, Type::Domain(_, b)) => a.is_compatible_with(b),
            
            _ => false,
        }
    }
    
    /// Unify two types for inference
    pub fn unify(&self, other: &Type) -> Option<Type> {
        match (self, other) {
            (Type::Infer, t) | (t, Type::Infer) => Some(t.clone()),
            (Type::Unknown, t) | (t, Type::Unknown) => Some(t.clone()),
            (a, b) if a == b => Some(a.clone()),
            
            // Numeric unification
            (Type::Int, Type::Float) | (Type::Float, Type::Int) => Some(Type::Float),
            
            // Optional unification
            (Type::Optional(a), Type::Optional(b)) => {
                a.unify(b).map(|t| Type::Optional(Box::new(t)))
            }
            (a, Type::Optional(b)) | (Type::Optional(b), a) => {
                a.unify(b).map(|t| Type::Optional(Box::new(t)))
            }
            
            // Uncertain unification
            (Type::Uncertain(a), Type::Uncertain(b)) => {
                a.unify(b).map(|t| Type::Uncertain(Box::new(t)))
            }
            (a, Type::Uncertain(b)) | (Type::Uncertain(b), a) => {
                a.unify(b).map(|t| Type::Uncertain(Box::new(t)))
            }
            
            _ => None,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Unit => write!(f, "()"),
            Type::Bool => write!(f, "bool"),
            Type::Int => write!(f, "int"),
            Type::Float => write!(f, "float"),
            Type::String => write!(f, "string"),
            Type::Array(t) => write!(f, "[{}]", t),
            Type::Tuple(types) => {
                write!(f, "({})", 
                    types.iter()
                        .map(|t| t.to_string())
                        .collect::<Vec<_>>()
                        .join(", "))
            }
            Type::Optional(t) => write!(f, "{}?", t),
            Type::Uncertain(t) => write!(f, "uncertain<{}>", t),
            Type::Embedding(dim) => write!(f, "embedding<{}>", dim),
            Type::Semantic(t) => write!(f, "semantic<{}>", t),
            Type::Function(ft) => {
                write!(f, "({}) -> {}", 
                    ft.params.iter()
                        .map(|t| t.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    ft.return_type)
            }
            Type::Struct(s) => write!(f, "{}", s.name),
            Type::Enum(e) => write!(f, "{}", e.name),
            Type::Interface(i) => write!(f, "{}", i.name),
            Type::Template(t) => write!(f, "template<{}>", t.name),
            Type::TypeParameter(name) => write!(f, "'{}", name),
            Type::Generic(name, args) => {
                write!(f, "{}<{}>", name,
                    args.iter()
                        .map(|t| t.to_string())
                        .collect::<Vec<_>>()
                        .join(", "))
            }
            Type::Any => write!(f, "any"),
            Type::Never => write!(f, "never"),
            Type::Unknown => write!(f, "unknown"),
            Type::Infer => write!(f, "_"),
            Type::Domain(domain, t) => write!(f, "{}::{}", domain, t),
        }
    }
}

/// Type environment for tracking variable types
#[derive(Debug, Clone)]
pub struct TypeEnvironment {
    scopes: Vec<HashMap<String, Type>>,
    type_definitions: HashMap<String, TypeDefinition>,
    current_function: Option<FunctionType>,
}

#[derive(Debug, Clone)]
pub enum TypeDefinition {
    Struct(StructType),
    Enum(EnumType),
    Interface(InterfaceType),
    TypeAlias(Type),
}

impl TypeEnvironment {
    pub fn new() -> Self {
        let mut env = Self {
            scopes: vec![HashMap::new()],
            type_definitions: HashMap::new(),
            current_function: None,
        };
        
        // Add built-in types
        env.add_builtins();
        env
    }
    
    fn add_builtins(&mut self) {
        // Add standard library types
        self.define_type("Result", TypeDefinition::Enum(EnumType {
            name: "Result".to_string(),
            variants: {
                let mut v = HashMap::new();
                v.insert("Ok".to_string(), Some(Type::TypeParameter("T".to_string())));
                v.insert("Err".to_string(), Some(Type::TypeParameter("E".to_string())));
                v
            },
            type_params: vec!["T".to_string(), "E".to_string()],
        }));
        
        self.define_type("Option", TypeDefinition::Enum(EnumType {
            name: "Option".to_string(),
            variants: {
                let mut v = HashMap::new();
                v.insert("Some".to_string(), Some(Type::TypeParameter("T".to_string())));
                v.insert("None".to_string(), None);
                v
            },
            type_params: vec!["T".to_string()],
        }));
    }
    
    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }
    
    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }
    
    pub fn define(&mut self, name: String, ty: Type) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, ty);
        }
    }
    
    pub fn lookup(&self, name: &str) -> Option<&Type> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
            }
        }
        None
    }
    
    pub fn define_type(&mut self, name: &str, def: TypeDefinition) {
        self.type_definitions.insert(name.to_string(), def);
    }
    
    pub fn lookup_type(&self, name: &str) -> Option<&TypeDefinition> {
        self.type_definitions.get(name)
    }
    
    pub fn set_current_function(&mut self, func: FunctionType) {
        self.current_function = Some(func);
    }
    
    pub fn clear_current_function(&mut self) {
        self.current_function = None;
    }
    
    pub fn current_return_type(&self) -> Option<&Type> {
        self.current_function.as_ref().map(|f| &*f.return_type)
    }
}

/// Type inference engine
pub struct TypeInference {
    env: TypeEnvironment,
    constraints: Vec<TypeConstraint>,
    substitutions: HashMap<String, Type>,
    next_type_var: usize,
}

#[derive(Debug, Clone)]
pub enum TypeConstraint {
    Equal(Type, Type),
    Subtype(Type, Type),
    HasField(Type, String, Type),
    HasMethod(Type, String, FunctionType),
    Implements(Type, String),
}

impl TypeInference {
    pub fn new() -> Self {
        Self {
            env: TypeEnvironment::new(),
            constraints: Vec::new(),
            substitutions: HashMap::new(),
            next_type_var: 0,
        }
    }
    
    pub fn fresh_type_var(&mut self) -> Type {
        let var = Type::TypeParameter(format!("T{}", self.next_type_var));
        self.next_type_var += 1;
        var
    }
    
    pub fn add_constraint(&mut self, constraint: TypeConstraint) {
        self.constraints.push(constraint);
    }
    
    pub fn solve(&mut self) -> Result<(), String> {
        while !self.constraints.is_empty() {
            let constraint = self.constraints.pop().unwrap();
            self.solve_constraint(constraint)?;
        }
        Ok(())
    }
    
    fn solve_constraint(&mut self, constraint: TypeConstraint) -> Result<(), String> {
        match constraint {
            TypeConstraint::Equal(t1, t2) => {
                if let Some(unified) = t1.unify(&t2) {
                    // Add substitution if needed
                    if let Type::TypeParameter(name) = &t1 {
                        self.substitutions.insert(name.clone(), unified);
                    } else if let Type::TypeParameter(name) = &t2 {
                        self.substitutions.insert(name.clone(), unified);
                    }
                } else {
                    return Err(format!("Cannot unify types {} and {}", t1, t2));
                }
            }
            
            TypeConstraint::Subtype(sub, sup) => {
                if !sub.is_compatible_with(&sup) {
                    return Err(format!("Type {} is not a subtype of {}", sub, sup));
                }
            }
            
            // Other constraint types...
            _ => {}
        }
        
        Ok(())
    }
    
    pub fn apply_substitutions(&self, ty: &Type) -> Type {
        match ty {
            Type::TypeParameter(name) => {
                self.substitutions.get(name).cloned().unwrap_or_else(|| ty.clone())
            }
            Type::Array(inner) => {
                Type::Array(Box::new(self.apply_substitutions(inner)))
            }
            Type::Optional(inner) => {
                Type::Optional(Box::new(self.apply_substitutions(inner)))
            }
            Type::Uncertain(inner) => {
                Type::Uncertain(Box::new(self.apply_substitutions(inner)))
            }
            // Apply to other compound types...
            _ => ty.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_type_compatibility() {
        let int = Type::Int;
        let float = Type::Float;
        let string = Type::String;
        
        assert!(int.is_compatible_with(&int));
        assert!(int.is_compatible_with(&float));
        assert!(!int.is_compatible_with(&string));
        
        let opt_int = Type::Optional(Box::new(Type::Int));
        assert!(int.is_compatible_with(&opt_int));
        assert!(opt_int.is_compatible_with(&int));
    }
    
    #[test]
    fn test_type_unification() {
        let infer = Type::Infer;
        let int = Type::Int;
        
        assert_eq!(infer.unify(&int), Some(int.clone()));
        
        let opt_int = Type::Optional(Box::new(Type::Int));
        let opt_float = Type::Optional(Box::new(Type::Float));
        
        assert_eq!(
            opt_int.unify(&opt_float),
            Some(Type::Optional(Box::new(Type::Float)))
        );
    }
    
    #[test]
    fn test_type_inference() {
        let mut inference = TypeInference::new();
        
        let t1 = inference.fresh_type_var();
        let t2 = Type::Int;
        
        inference.add_constraint(TypeConstraint::Equal(t1.clone(), t2.clone()));
        inference.solve().unwrap();
        
        let result = inference.apply_substitutions(&t1);
        assert_eq!(result, Type::Int);
    }
}