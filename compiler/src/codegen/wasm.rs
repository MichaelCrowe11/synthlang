/*!
 * WebAssembly Code Generator for SYNTH
 * Compiles SYNTH AST to WebAssembly bytecode
 */

use crate::ast::{self, Node, Expression, Statement, Declaration, Program};
use anyhow::{Result, anyhow};
use wasm_encoder::{
    Module, TypeSection, FunctionSection, ExportSection, CodeSection,
    ImportSection, MemorySection, GlobalSection, DataSection,
    Function as WasmFunction, Instruction, ValType, BlockType,
    ExportKind, MemoryType, GlobalType,
};

pub struct WasmGenerator {
    module: Module,
    type_section: TypeSection,
    function_section: FunctionSection,
    export_section: ExportSection,
    code_section: CodeSection,
    import_section: ImportSection,
    memory_section: MemorySection,
    global_section: GlobalSection,
    data_section: DataSection,
    
    // State tracking
    function_types: Vec<FunctionSignature>,
    local_vars: Vec<LocalVariable>,
    string_constants: Vec<String>,
    current_function: Option<WasmFunction>,
}

#[derive(Debug, Clone)]
struct FunctionSignature {
    params: Vec<ValType>,
    results: Vec<ValType>,
}

#[derive(Debug, Clone)]
struct LocalVariable {
    name: String,
    ty: ValType,
    index: u32,
}

impl WasmGenerator {
    pub fn new() -> Self {
        Self {
            module: Module::new(),
            type_section: TypeSection::new(),
            function_section: FunctionSection::new(),
            export_section: ExportSection::new(),
            code_section: CodeSection::new(),
            import_section: ImportSection::new(),
            memory_section: MemorySection::new(),
            global_section: GlobalSection::new(),
            data_section: DataSection::new(),
            function_types: Vec::new(),
            local_vars: Vec::new(),
            string_constants: Vec::new(),
            current_function: None,
        }
    }
    
    pub fn generate(&mut self, program: &Program) -> Result<Vec<u8>> {
        // Setup memory for heap allocation
        self.setup_memory();
        
        // Import AI runtime functions
        self.import_ai_functions();
        
        // Process all declarations
        for item in &program.items {
            self.generate_declaration(item)?;
        }
        
        // Finalize module
        self.finalize_module()
    }
    
    fn setup_memory(&mut self) {
        // Create linear memory with 1 initial page (64KB)
        self.memory_section.memory(MemoryType {
            minimum: 1,
            maximum: Some(256), // Max 16MB
            memory64: false,
            shared: false,
        });
        
        // Export memory for JavaScript access
        self.export_section.export("memory", ExportKind::Memory, 0);
    }
    
    fn import_ai_functions(&mut self) {
        // Import AI generation function
        let ai_generate_type = self.add_function_type(
            vec![ValType::I32, ValType::I32], // prompt ptr, prompt len
            vec![ValType::I32, ValType::I32], // result ptr, result len
        );
        
        self.import_section.import(
            "synth",
            "ai_generate",
            wasm_encoder::EntityType::Function(ai_generate_type),
        );
        
        // Import embedding function
        let embed_type = self.add_function_type(
            vec![ValType::I32, ValType::I32], // text ptr, text len
            vec![ValType::I32], // embedding ptr
        );
        
        self.import_section.import(
            "synth",
            "embed",
            wasm_encoder::EntityType::Function(embed_type),
        );
        
        // Import similarity function
        let similarity_type = self.add_function_type(
            vec![ValType::I32, ValType::I32], // vec1 ptr, vec2 ptr
            vec![ValType::F32], // similarity score
        );
        
        self.import_section.import(
            "synth",
            "similarity",
            wasm_encoder::EntityType::Function(similarity_type),
        );
    }
    
    fn add_function_type(&mut self, params: Vec<ValType>, results: Vec<ValType>) -> u32 {
        let signature = FunctionSignature {
            params: params.clone(),
            results: results.clone(),
        };
        
        // Check if type already exists
        for (i, existing) in self.function_types.iter().enumerate() {
            if existing.params == signature.params && existing.results == signature.results {
                return i as u32;
            }
        }
        
        // Add new type
        let index = self.function_types.len() as u32;
        self.function_types.push(signature);
        self.type_section.function(params, results);
        index
    }
    
    fn generate_declaration(&mut self, decl: &Declaration) -> Result<()> {
        match decl {
            Declaration::Function(func) => self.generate_function(func),
            Declaration::Variable(var) => self.generate_global_variable(var),
            Declaration::Constant(const_decl) => self.generate_global_constant(const_decl),
            _ => Ok(()), // Skip other declarations for now
        }
    }
    
    fn generate_function(&mut self, func: &ast::Function) -> Result<()> {
        // Create function signature
        let param_types: Vec<ValType> = func.parameters.iter()
            .map(|_| ValType::I32) // Simplification: all params are i32 for now
            .collect();
        
        let return_type = if func.return_type.is_some() {
            vec![ValType::I32]
        } else {
            vec![]
        };
        
        let type_index = self.add_function_type(param_types.clone(), return_type);
        self.function_section.function(type_index);
        
        // Start function body
        let mut function = WasmFunction::new(vec![]); // No additional locals for now
        self.current_function = Some(function);
        
        // Setup local variables for parameters
        self.local_vars.clear();
        for (i, param) in func.parameters.iter().enumerate() {
            self.local_vars.push(LocalVariable {
                name: param.name.value.clone(),
                ty: ValType::I32,
                index: i as u32,
            });
        }
        
        // Generate function body
        self.generate_block(&func.body)?;
        
        // Add implicit return if needed
        if func.return_type.is_none() {
            self.emit_instruction(Instruction::End);
        }
        
        // Add function to code section
        if let Some(function) = self.current_function.take() {
            self.code_section.function(&function);
        }
        
        // Export main function
        if func.name.value == "main" {
            let func_index = self.function_section.len() - 1;
            self.export_section.export("main", ExportKind::Func, func_index);
        }
        
        Ok(())
    }
    
    fn generate_global_variable(&mut self, var: &ast::Variable) -> Result<()> {
        // For simplicity, store globals as i32
        self.global_section.global(
            GlobalType {
                val_type: ValType::I32,
                mutable: !var.is_const,
            },
            &Instruction::I32Const(0), // Initial value
        );
        
        Ok(())
    }
    
    fn generate_global_constant(&mut self, const_decl: &ast::Constant) -> Result<()> {
        self.global_section.global(
            GlobalType {
                val_type: ValType::I32,
                mutable: false,
            },
            &Instruction::I32Const(0), // Initial value
        );
        
        Ok(())
    }
    
    fn generate_block(&mut self, block: &ast::Block) -> Result<()> {
        for statement in &block.statements {
            self.generate_statement(statement)?;
        }
        Ok(())
    }
    
    fn generate_statement(&mut self, stmt: &Statement) -> Result<()> {
        match stmt {
            Statement::Expression(expr) => {
                self.generate_expression(expr)?;
                // Drop result if not used
                self.emit_instruction(Instruction::Drop);
                Ok(())
            }
            
            Statement::Return(expr) => {
                if let Some(e) = expr {
                    self.generate_expression(e)?;
                }
                self.emit_instruction(Instruction::Return);
                Ok(())
            }
            
            Statement::If { condition, then_block, else_block } => {
                self.generate_expression(condition)?;
                
                if let Some(else_b) = else_block {
                    self.emit_instruction(Instruction::If(BlockType::Empty));
                    self.generate_block(then_block)?;
                    self.emit_instruction(Instruction::Else);
                    self.generate_block(else_b)?;
                    self.emit_instruction(Instruction::End);
                } else {
                    self.emit_instruction(Instruction::If(BlockType::Empty));
                    self.generate_block(then_block)?;
                    self.emit_instruction(Instruction::End);
                }
                
                Ok(())
            }
            
            Statement::While { condition, body } => {
                self.emit_instruction(Instruction::Loop(BlockType::Empty));
                self.generate_expression(condition)?;
                self.emit_instruction(Instruction::I32Eqz);
                self.emit_instruction(Instruction::BrIf(1)); // Break if condition is false
                self.generate_block(body)?;
                self.emit_instruction(Instruction::Br(0)); // Continue loop
                self.emit_instruction(Instruction::End);
                
                Ok(())
            }
            
            Statement::Break => {
                self.emit_instruction(Instruction::Br(1)); // Break out of loop
                Ok(())
            }
            
            Statement::Continue => {
                self.emit_instruction(Instruction::Br(0)); // Continue to loop start
                Ok(())
            }
            
            _ => Ok(()), // Other statements not implemented yet
        }
    }
    
    fn generate_expression(&mut self, expr: &Node<Expression>) -> Result<()> {
        match &expr.value {
            Expression::Literal(lit) => self.generate_literal(lit),
            
            Expression::Identifier(name) => {
                // Look up local variable
                if let Some(local) = self.local_vars.iter().find(|v| &v.name == name) {
                    self.emit_instruction(Instruction::LocalGet(local.index));
                } else {
                    return Err(anyhow!("Undefined variable: {}", name));
                }
                Ok(())
            }
            
            Expression::Binary { op, left, right } => {
                self.generate_expression(left)?;
                self.generate_expression(right)?;
                self.generate_binary_op(op)?;
                Ok(())
            }
            
            Expression::Unary { op, operand } => {
                self.generate_expression(operand)?;
                self.generate_unary_op(op)?;
                Ok(())
            }
            
            Expression::Call { function, arguments } => {
                // Generate arguments
                for arg in arguments {
                    self.generate_expression(arg)?;
                }
                
                // Call function (simplified: assume function index)
                if let Expression::Identifier(name) = &function.value {
                    if name == "print" {
                        // Special case for print (would be imported)
                        self.emit_instruction(Instruction::Call(0));
                    } else {
                        // Look up function index
                        self.emit_instruction(Instruction::Call(0)); // Placeholder
                    }
                }
                
                Ok(())
            }
            
            Expression::AiGenerate { prompt, .. } => {
                // Generate prompt string
                self.generate_expression(prompt)?;
                // Call imported AI generate function
                self.emit_instruction(Instruction::Call(0)); // Index of imported ai_generate
                Ok(())
            }
            
            Expression::SemanticSimilarity { left, right } => {
                self.generate_expression(left)?;
                self.generate_expression(right)?;
                // Call imported similarity function
                self.emit_instruction(Instruction::Call(2)); // Index of imported similarity
                Ok(())
            }
            
            _ => Ok(()), // Other expressions not implemented yet
        }
    }
    
    fn generate_literal(&mut self, lit: &ast::Literal) -> Result<()> {
        match lit {
            ast::Literal::Integer(n) => {
                self.emit_instruction(Instruction::I32Const(*n as i32));
            }
            ast::Literal::Float(f) => {
                self.emit_instruction(Instruction::F32Const(*f as f32));
            }
            ast::Literal::Boolean(b) => {
                self.emit_instruction(Instruction::I32Const(if *b { 1 } else { 0 }));
            }
            ast::Literal::String(s) => {
                // Store string in memory and return pointer
                let ptr = self.allocate_string(s);
                self.emit_instruction(Instruction::I32Const(ptr as i32));
            }
            ast::Literal::Null => {
                self.emit_instruction(Instruction::I32Const(0));
            }
        }
        Ok(())
    }
    
    fn generate_binary_op(&mut self, op: &ast::BinaryOperator) -> Result<()> {
        use ast::BinaryOperator::*;
        
        let instruction = match op {
            Add => Instruction::I32Add,
            Subtract => Instruction::I32Sub,
            Multiply => Instruction::I32Mul,
            Divide => Instruction::I32DivS,
            Modulo => Instruction::I32RemS,
            Equal => Instruction::I32Eq,
            NotEqual => Instruction::I32Ne,
            Less => Instruction::I32LtS,
            LessEqual => Instruction::I32LeS,
            Greater => Instruction::I32GtS,
            GreaterEqual => Instruction::I32GeS,
            And => Instruction::I32And,
            Or => Instruction::I32Or,
            _ => return Ok(()), // Other operators not implemented
        };
        
        self.emit_instruction(instruction);
        Ok(())
    }
    
    fn generate_unary_op(&mut self, op: &ast::UnaryOperator) -> Result<()> {
        use ast::UnaryOperator::*;
        
        match op {
            Not => {
                self.emit_instruction(Instruction::I32Eqz);
            }
            Negate => {
                self.emit_instruction(Instruction::I32Const(0));
                self.emit_instruction(Instruction::I32Sub);
            }
        }
        
        Ok(())
    }
    
    fn allocate_string(&mut self, s: &str) -> u32 {
        // Simple string allocation in data section
        let offset = self.string_constants.len() as u32 * 1024; // Each string gets 1KB
        self.string_constants.push(s.to_string());
        
        // Add to data section
        self.data_section.active(
            0, // Memory index
            &Instruction::I32Const(offset as i32),
            s.as_bytes().to_vec(),
        );
        
        offset
    }
    
    fn emit_instruction(&mut self, instruction: Instruction) {
        if let Some(ref mut function) = self.current_function {
            function.instruction(&instruction);
        }
    }
    
    fn finalize_module(&mut self) -> Result<Vec<u8>> {
        // Add sections to module in correct order
        let mut module = Module::new();
        
        if !self.type_section.is_empty() {
            module.section(&self.type_section);
        }
        if !self.import_section.is_empty() {
            module.section(&self.import_section);
        }
        if !self.function_section.is_empty() {
            module.section(&self.function_section);
        }
        if !self.memory_section.is_empty() {
            module.section(&self.memory_section);
        }
        if !self.global_section.is_empty() {
            module.section(&self.global_section);
        }
        if !self.export_section.is_empty() {
            module.section(&self.export_section);
        }
        if !self.code_section.is_empty() {
            module.section(&self.code_section);
        }
        if !self.data_section.is_empty() {
            module.section(&self.data_section);
        }
        
        Ok(module.finish())
    }
}

// Helper trait implementations
impl TypeSection {
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl ImportSection {
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl FunctionSection {
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl ExportSection {
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl CodeSection {
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl MemorySection {
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl GlobalSection {
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl DataSection {
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_wasm_generation_empty_program() {
        let mut generator = WasmGenerator::new();
        let program = Program { items: vec![] };
        
        let result = generator.generate(&program);
        assert!(result.is_ok());
        
        let wasm = result.unwrap();
        assert!(!wasm.is_empty());
        
        // Verify WASM magic number
        assert_eq!(&wasm[0..4], b"\0asm");
        // Verify version
        assert_eq!(&wasm[4..8], &[1, 0, 0, 0]);
    }
}