package org.clulab.scala_transformers.tokenizer.jni;

public class JavaJniTokenization {
    public int tokenIds[];
    public int wordIds[];
    public String tokens[];

    public JavaJniTokenization(int[] tokenIds, int[] wordIds, String[] tokens) {
        this.tokenIds = tokenIds;
        this.wordIds = wordIds;
        this.tokens = tokens;
    }
}
