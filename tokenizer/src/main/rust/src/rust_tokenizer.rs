use jni::JNIEnv;
use jni::objects::JClass;
use jni::objects::JObject;
use jni::objects::JString;
use jni::objects::JValue;
use jni::sys::jlong;
use jni::sys::jobject;
use jni::sys::jsize;
use jni::sys::jobjectArray;
use tokenizers::tokenizer::Tokenizer;

fn create_tokenizer(name: &String) -> i64 {
    // See https://doc.rust-lang.org/std/primitive.pointer.html.
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

#[no_mangle]
pub extern "system" fn Java_org_clulab_scala_1transformers_tokenizer_jni_JavaJniTokenizer_native_1create(env: JNIEnv,
        _class: JClass, j_name: JString) -> jlong {
    let r_name: String = env.get_string(j_name).unwrap().into();
    eprintln!("[Tokenizer] => create_rust_tokenizer(\"{}\")", r_name);

    let tokenizer_id = create_tokenizer(&r_name);
    eprintln!("[Tokenizer] <= {}", tokenizer_id);
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
