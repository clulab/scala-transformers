use jni::JNIEnv;
use jni::objects::JClass;
use jni::objects::JObject;
use jni::objects::JString;
use jni::objects::JValue;
use jni::sys::jbyteArray;
use jni::sys::jlong;
use jni::sys::jobject;
use jni::sys::jsize;
use jni::sys::jobjectArray;
use std::path::Path;
use tokenizers::tokenizer::{Result, Tokenizer};

fn create_from_pretrained(name: &String) -> Result<i64> {
    // See https://doc.rust-lang.org/std/primitive.pointer.html.
    let tokenizer_result: Result<Tokenizer> = Tokenizer::from_pretrained(name, None);
    let result: Result<i64> = tokenizer_result.map(|tokenizer_stack| {
        let tokenizer_heap: Box<Tokenizer> = Box::new(tokenizer_stack);
        let tokenizer_ref: &'static mut Tokenizer = Box::leak(tokenizer_heap);
        let tokenizer_ptr: *mut Tokenizer = tokenizer_ref;
        let tokenizer_id: i64 = tokenizer_ptr as i64;
    
        tokenizer_id
    });

    return result;
}

fn create_from_file(file_name: &String) -> Result<i64> {
    // See https://doc.rust-lang.org/std/primitive.pointer.html.
    let tokenizer_result: Result<Tokenizer> = Tokenizer::from_file(Path::new(file_name));
    let result: Result<i64> = tokenizer_result.map(|tokenizer_stack| {
        let tokenizer_heap: Box<Tokenizer> = Box::new(tokenizer_stack);
        let tokenizer_ref: &'static mut Tokenizer = Box::leak(tokenizer_heap);
        let tokenizer_ptr: *mut Tokenizer = tokenizer_ref;
        let tokenizer_id: i64 = tokenizer_ptr as i64;

        tokenizer_id
    });

    return result;
}

fn create_from_bytes(bytes_vec: &Vec<u8>) -> Result<i64> {
    let bytes_ref = bytes_vec.as_slice();
    let tokenizer_result = Tokenizer::from_bytes(bytes_ref);
    let result: Result<i64> = tokenizer_result.map(|tokenizer_stack| {
        let tokenizer_heap: Box<Tokenizer> = Box::new(tokenizer_stack);
        let tokenizer_ref: &'static mut Tokenizer = Box::leak(tokenizer_heap);
        let tokenizer_ptr: *mut Tokenizer = tokenizer_ref;
        let tokenizer_id: i64 = tokenizer_ptr as i64;

        tokenizer_id
    });

    return result;
}

fn destroy_tokenizer(tokenizer_id: i64) {
    if tokenizer_id == 0 {
        return;
    }
    let tokenizer_ptr = tokenizer_id as *mut Tokenizer;
    // This takes ownership and will cause memory to be released.
    unsafe { let _ = Box::from_raw(tokenizer_ptr); };
    return;
}

#[no_mangle]
pub extern "system" fn Java_org_clulab_scala_1transformers_tokenizer_jni_JavaJniTokenizer_native_1create_1from_1pretrained(env: JNIEnv,
        _class: JClass, j_name: JString) -> jlong {
    let r_name: String = env.get_string(j_name).unwrap().into();
    eprintln!("[Tokenizer] => create_from_pretrained(\"{}\")", r_name);

    let tokenizer_result = create_from_pretrained(&r_name);
    let tokenizer_id = match tokenizer_result {
        Ok(tokenizer_id) => {
            eprintln!("[Tokenizer] <= {}", tokenizer_id);
            tokenizer_id
        },
        Err(error) => {
            eprintln!("[Tokenizer] <= {}", error);
            0 as i64
        }
    };

    return tokenizer_id;
}

#[no_mangle]
pub extern "system" fn Java_org_clulab_scala_1transformers_tokenizer_jni_JavaJniTokenizer_native_1create_1from_1file(env: JNIEnv,
        _class: JClass, j_file_name: JString) -> jlong {
    let r_file_name: String = env.get_string(j_file_name).unwrap().into();
    eprintln!("[Tokenizer] => create_from_file(\"{}\")", r_file_name);

    let tokenizer_result = create_from_file(&r_file_name);
    let tokenizer_id = match tokenizer_result {
        Ok(tokenizer_id) => {
            eprintln!("[Tokenizer] <= {}", tokenizer_id);
            tokenizer_id
        },
        Err(error) => {
            eprintln!("[Tokenizer] <= {}", error);
            0 as i64
        }
    };
    
    return tokenizer_id;
}

#[no_mangle]
pub extern "system" fn Java_org_clulab_scala_1transformers_tokenizer_jni_JavaJniTokenizer_native_1create_1from_1bytes(env: JNIEnv,
        _class: JClass, j_bytes: jbyteArray) -> jlong {
    eprintln!("[Tokenizer] => create_from_bytes(...)");
    let buffer = env.convert_byte_array(j_bytes).unwrap();
    // let buffer: Vec<u8> = env.convert_byte_array(j_bytes).unwrap();
    // let buffer = env.get_byte_array_elements(j_bytes, ReleaseMode::NoCopyBack).unwrap();

    let tokenizer_result = create_from_bytes(&buffer);
    let tokenizer_id = match tokenizer_result {
        Ok(tokenizer_id) => {
            eprintln!("[Tokenizer] <= {}", tokenizer_id);
            tokenizer_id
        },
        Err(error) => {
            eprintln!("[Tokenizer] <= {}", error);
            0 as i64
        }
    };

    return tokenizer_id;
}


#[no_mangle]
pub extern "system" fn Java_org_clulab_scala_1transformers_tokenizer_jni_JavaJniTokenizer_native_1destroy(_env: JNIEnv,
        _class: JClass, tokenizer_id: jlong) -> () {
    eprintln!("[Tokenizer] => destroy_rust_tokenizer({})", tokenizer_id);
    
    destroy_tokenizer(tokenizer_id);
}

#[no_mangle]
pub extern "system" fn Java_org_clulab_scala_1transformers_tokenizer_jni_JavaJniTokenizer_native_1tokenize(env: JNIEnv,
        _class: JClass, tokenizer_id: jlong, j_words: jobjectArray) -> jobject {
    // eprintln!("[Tokenizer] => rust_tokenizer_tokenize({}, <words>)", tokenizer_id);

    let word_count = env.get_array_length(j_words).unwrap();
    let mut r_words: Vec<String> = Vec::with_capacity(word_count as usize); 
    for i in 0..word_count {
        let j_word = JString::from(env.get_object_array_element(j_words, i).unwrap());
        let r_word: String = env.get_string(j_word).unwrap().into();

        // eprintln!("{} = {}", i, r_word);
        r_words.push(r_word);
    }

    let tokenizer = unsafe { &*(tokenizer_id as *const Tokenizer) };
    let encoding = tokenizer.encode(&r_words[..], true).unwrap();
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
    
    let tokens_count = tokens.len();
    let j_token_ids = env.new_int_array(tokens_count as i32).unwrap();
    let _ = env.set_int_array_region(j_token_ids, 0, token_id_i32s);
    let j_word_ids = env.new_int_array(tokens_count as i32).unwrap();
    let _ = env.set_int_array_region(j_word_ids, 0, word_id_i32s);

    let j_tokens = env.new_object_array(tokens.len() as jsize, "java/lang/String", JObject::null()).unwrap();
    for i in 0..tokens_count {
        let token = &tokens[i];
        let j_token = env.new_string(token).unwrap();
        let _ = env.set_object_array_element(j_tokens, i as jsize, j_token);

        // println!("{}", token);
    }

    let j_tokenization = env.new_object(
        "org/clulab/scala_transformers/tokenizer/jni/JavaJniTokenization",
        "([I[I[Ljava/lang/String;)V",
        &[
            JValue::from(j_token_ids),
            JValue::from(j_word_ids),
            JValue::from(j_tokens)
        ]
    ).unwrap().into_inner();

    return j_tokenization;
}
