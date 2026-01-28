use std::path::Path;
use tokenizers::tokenizer::{Result, Tokenizer}; // , EncodeInput};

fn main() -> Result<()> {
    let path = Path::new("./src/microsoft/deberta-v3-base/tokenizer.json");
    let tokenizer = Tokenizer::from_file(path)?; 
    let text = "Hello world, this is a Rust tokenizer example!";
    let encoding = tokenizer.encode(text, true)?;

    println!("Original text: {}", text);
    println!("Token IDs: {:?}", encoding.get_ids());
    println!("Word IDs: {:?}", encoding.get_word_ids());
    println!("Tokens: {:?}", encoding.get_tokens());

    Ok(())
}

