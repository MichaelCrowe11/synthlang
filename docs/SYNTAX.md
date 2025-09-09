# SYNTH Language Syntax Specification
## Core Grammar for Proof-of-Concept Implementation

### Basic Types
```synth
// Primitive types
let number: int = 42
let text: string = "Hello, SYNTH!"
let flag: bool = true
let decimal: float = 3.14159

// AI-native types
let embedding: vector<768> = embed("AI research")
let uncertainty: uncertain<int> = 75 @ 0.85  // value @ confidence
let probability: prob<bool> = maybe(0.7)     // 70% true
```

### Variables and Constants
```synth
// Mutable variables
var counter = 0
var name: string = "SYNTH"

// Immutable values
let pi = 3.14159
const MAX_ITERATIONS = 1000
```

### Functions
```synth
// Basic functions
function add(a: int, b: int) -> int {
    return a + b
}

// AI-enhanced functions
@ai_model("gpt-4")
function generate_summary(text: string) -> string {
    prompt: "Summarize the following text: {text}"
}

// Uncertain return types
function risky_operation() -> uncertain<int> {
    return compute_result() @ confidence_level()
}
```

### AI Operations
```synth
// LLM calls
let response = ai.generate("Write a haiku about programming")

// Embeddings and similarity
let query_vec = embed("machine learning")
let doc_vec = embed("artificial intelligence") 
let similarity = query_vec ~~ doc_vec  // semantic similarity operator

// Vector operations
let combined = query_vec + doc_vec
let normalized = query_vec.normalize()
```

### Control Flow
```synth
// If statements
if condition {
    action()
} else if other_condition {
    other_action()
} else {
    default_action()
}

// Loops
for i in 0..10 {
    print(i)
}

while condition {
    work()
}

// Pattern matching
match result {
    Success(value) => process(value),
    Error(msg) => handle_error(msg),
    Uncertain(val, conf) if conf > 0.8 => trust(val),
    _ => ask_human()
}
```

### AI-Native Constructs
```synth
// Semantic queries
let results = knowledge_graph.find {
    ?entity hasProperty "programming_language"
    ?entity similarity ~~ embed("functional") > 0.7
}

// Probabilistic programming
uncertain block {
    let outcome = flip_coin() @ 0.6  // biased coin
    if outcome {
        return "heads" @ 0.6
    } else {
        return "tails" @ 0.4  
    }
}

// Neural computation
@neural_network
function classify(input: tensor<784>) -> prob<string> {
    hidden = relu(input @ weights.hidden)
    output = softmax(hidden @ weights.output)
    return output
}
```

### Cross-Domain Integration
```synth
// Healthcare to Finance domain bridge
function assess_insurance_risk(
    health_data: HealthRecord,
    financial_data: CreditProfile
) -> InsuranceRisk {
    // Transform health metrics to risk factors
    health_risk = health_data ~~ risk_patterns
    
    // Cross-domain reasoning
    combined_risk = bridge(health_risk, financial_data.risk_score)
    
    return InsuranceRisk {
        level: combined_risk.level,
        confidence: combined_risk.confidence,
        explanation: ai.explain(combined_risk)
    }
}
```

### Templates (Liquid++ Evolution)
```synth
template ProductCard {
    @ai_enhanced
    render(product: Product) {
        <div class="product">
            <h2>{product.name}</h2>
            <p>{ai.enhance_description(product.description)}</p>
            <price>{product.price | format_currency}</price>
            
            // Semantic recommendations
            <related>
                {for item in products where item ~~ product > 0.8}
                    <ProductThumbnail data={item} />
                {/for}
            </related>
        </div>
    }
}
```

## Lexical Tokens

### Keywords
```
function, let, var, const, if, else, while, for, in, match, return,
true, false, template, render, ai, embed, uncertain, prob, @
```

### AI-Specific Tokens
```
~~   (semantic similarity)
@    (confidence/uncertainty)  
|>   (pipeline operator)
<|   (reverse pipeline)
```

### Operators
```
+, -, *, /, %, =, ==, !=, <, >, <=, >=, &&, ||, !, &, |, ^, <<, >>
```

### Delimiters
```
{, }, [, ], (, ), ,, ;, :, ., ->, =>, ::
```

### Literals
```
// Numbers
42, 3.14159, 0xFF, 0b1010, 1_000_000

// Strings  
"hello", 'world', """multi
line string"""

// Templates
`Hello {name}!`
```

## Grammar Rules (EBNF-style)

```ebnf
program = item*

item = function_def | variable_def | template_def

function_def = annotation* "function" IDENT "(" params? ")" return_type? block

variable_def = ("let" | "var" | "const") IDENT type_annotation? "=" expression

template_def = "template" IDENT block

block = "{" statement* "}"

statement = expression ";" | if_stmt | while_stmt | for_stmt | match_stmt | return_stmt

expression = ai_expr | binary_expr | unary_expr | call_expr | primary

ai_expr = "ai" "." IDENT "(" args? ")"
        | "embed" "(" expression ")"
        | expression "~~" expression
        | expression "@" expression

binary_expr = expression binary_op expression
unary_expr = unary_op expression
call_expr = expression "(" args? ")"

primary = IDENT | literal | "(" expression ")"

literal = NUMBER | STRING | BOOLEAN
```

This grammar defines our core language features while keeping it simple enough for a proof-of-concept implementation.