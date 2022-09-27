package org.clulab.transformers.tokenizer.j4rs;

import java.util.AbstractList;

public class JavaJ4rsTokenization {
    public int tokenIds[];
    public int wordIds[];
    public String tokens[];

    public JavaJ4rsTokenization(AbstractList<Integer> tokenIds, AbstractList<Integer> wordIds, AbstractList<String> tokens) {
        this.tokenIds = tokenIds.stream().mapToInt(i -> i).toArray();
        this.wordIds = wordIds.stream().mapToInt(i -> i).toArray();
        this.tokens = (String[]) tokens.toArray();
    }
}
