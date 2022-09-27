package org.clulab.transformers.tokenizer.j4rs

object TimeTokenizerApp extends App {
  val sentences = Array(
    Array("EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "."),
    Array("The", "Computational", "Language", "Understanding", "(", "CLU", ")", "Lab", "at", "University", "of", "Arizona", "is", "a", "team", "of", "faculty,", "students,", "and", "research", "programmers", "who", "work", "together", "to", "build", "systems", "that", "extract", "meaning", "from", "natural", "language", "texts", ",", "including", "question", "answering", "(", "answering", "natural", "language", "questions", ")", ",", "information", "extraction", "(", "extracting", "specific", "relations", "and", "events", ")", ",", "semantic", "role", "labeling", "(", "extracting", "semantic", "frames", "that", "model", "who", "did", "what", "to", "whom,", "when", "and", "where", "),", "parsing", "the", "discourse", "structure", "of", "complex", "texts,", "and", "other", "computational", "linguistics", "problems", "."),
    Array("These", "systems", "were", "used", "in", "several", "applications", ",", "ranging", "from", "extracting", "cancer", "signaling", "pathways", "from", "biomedical", "articles", "to", "automated", "systems", "for", "answering", "multiple-choice", "science-exam", "questions", "."),
    Array("The", "CLU", "lab", "includes", "members", "from", "the", "Computer", "Science", "department,", "the", "Linguistics", "department,", "and", "the", "School", "of", "Information", ".", "For", "more", "on", "natural", "language", "processing", "(", "NLP", ")", "work", "at", "UofA", ",", "please", "see", "our", "NLP", "cluster", "page", ".")
  )
  val name = "distilbert-base-cased"
  val tokenizer = ScalaJ4rsTokenizer(name)

  def loop(): Unit = {
    1.until(1000).par.foreach { _ =>
      sentences.foreach { words =>
        val tokenization = tokenizer.tokenize(words)
        // println(tokenization.tokens.mkString(" "))
      }
    }
  }

  def measure(): Unit = {
    val startTime = System.currentTimeMillis() / 1000.0
    loop()
    val endTime = System.currentTimeMillis() / 1000.0
    print(s"--- ${endTime - startTime} seconds ---")
  }

  measure()
}
