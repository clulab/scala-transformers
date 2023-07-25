package org.clulab.scala_transformers.tokenizer.jni;

public class JavaJniTokenizer {
    // Take the name and create a RustTokenizer.
    // return an instance
    private static native long native_create(String name);

    private static native long native_deserialize(String content);

    // Garbage collect the RustTokenizer.
    public static native void native_destroy(long rustTokenizerId);

    // Perform tokenization on the words.
    private static native JavaJniTokenization native_tokenize(long rustTokenizerId, String[] words);

    public static long create(String name) {
        long tokenizer_id = native_create(name);

        return tokenizer_id;
    }

    public static long deserialize(String content) {
        long tokenizer_id = native_deserialize(content);

        return tokenizer_id;
    }

    public static void destroy(long tokenizer_id) {
        native_destroy(tokenizer_id);
    };

    public static JavaJniTokenization tokenize(long tokenizer_id, String[] words) {
        JavaJniTokenization tokenization = native_tokenize(tokenizer_id, words);

        return tokenization;
    }
}
