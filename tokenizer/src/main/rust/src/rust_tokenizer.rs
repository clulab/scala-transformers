use j4rs::InvocationArg;
use j4rs::prelude::*;
use j4rs_derive::*;
use std::convert::TryFrom;
use std::result::Result;
use tokenizers::tokenizer::Tokenizer;

fn create_tokenizer(name: &String) -> i64 {
    // See https://doc.rust-lang.org/std/primitive.pointer.html
    let tokenizer_stack: Tokenizer = Tokenizer::from_pretrained(name, None).unwrap();
    let tokenizer_heap: Box<Tokenizer> = Box::new(tokenizer_stack);
    let tokenizer_ref: &'static mut Tokenizer = Box::leak(tokenizer_heap);
    let tokenizer_ptr: *mut Tokenizer = tokenizer_ref;
    let tokenizer_id: i64 = tokenizer_ptr as i64;

    return tokenizer_id;
}

fn destroy_tokenizer(tokenizer_id: i64) {
    let tokenizer_ptr = tokenizer_id as *mut Tokenizer;
    // This takes ownership and will cause memory to be released.
    unsafe { Box::from_raw(tokenizer_ptr) };
    return;
}

#[call_from_java("org.clulab.transformers.tokenizer.j4rs.JavaJ4rsTokenizer.create")]
fn create_rust_tokenizer(name_instance: Instance) -> Result<Instance, String> {
    let jvm: Jvm = Jvm::attach_thread().unwrap();
    let name: String = jvm.to_rust(name_instance).unwrap();
    println!("create_rust_tokenizer(\"{}\")", name);

    let tokenizer_id = create_tokenizer(&name);
    let tokenizer_id_invocation_arg = InvocationArg::try_from(tokenizer_id).unwrap();
    let tokenizer_id_result = Instance::try_from(tokenizer_id_invocation_arg).map_err(|error| format!("{}", error));

    return tokenizer_id_result;
}

#[call_from_java("org.clulab.transformers.tokenizer.j4rs.JavaJ4rsTokenizer.destroy")]
fn destroy_rust_tokenizer(tokenizer_id_instance: Instance) {
    let jvm: Jvm = Jvm::attach_thread().unwrap();
    let tokenizer_id: i64 = jvm.to_rust(tokenizer_id_instance).unwrap();
    println!("destroy_rust_tokenizer({})", tokenizer_id);

    destroy_tokenizer(tokenizer_id);
    return;
}

#[call_from_java("org.clulab.transformers.tokenizer.j4rs.JavaJ4rsTokenizer.tokenize")]
fn rust_tokenizer_tokenize(tokenizer_id_instance: Instance, words_instance: Instance) -> Result<Instance, String> {
    let jvm: Jvm = Jvm::attach_thread().unwrap();
    let tokenizer_id: i64 = jvm.to_rust(tokenizer_id_instance).unwrap();
    let words: Vec<String> = jvm.to_rust(words_instance).unwrap();
    // println!("rust_tokenizer_tokenize({}, <words>)", tokenizer_id);

    let tokenizer = unsafe { &*(tokenizer_id as *const Tokenizer) };
    let encoding = tokenizer.encode(&words[..], true).unwrap();
    let token_id_u32s = encoding.get_ids();
    let token_id_i32s = unsafe { std::mem::transmute::<&[u32], &[i32]>(token_id_u32s) };
    let word_id_opts = encoding.get_word_ids();
    let word_id_i32s = &word_id_opts
       .iter()
       .map(|&word_id_opt| {
           if word_id_opt.is_some() {
               word_id_opt.unwrap() as i32
           } else {
               -1
           }
       })
       .collect::<Vec<_>>()[..];
    let tokens = encoding.get_tokens();
    let java_tokenization_instance = jvm.create_instance("org.clulab.transformers.tokenizer.j4rs.JavaJ4rsTokenization", &[
        InvocationArg::try_from(token_id_i32s).unwrap(),
        InvocationArg::try_from(word_id_i32s).unwrap(),
        InvocationArg::try_from(tokens).unwrap()
    ]).unwrap();

    return Ok(java_tokenization_instance);
}
