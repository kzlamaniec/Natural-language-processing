# załadowanie bibliotek
library(tm)
library(hunspell)

## Tworzenie korpusu i wstępne przetwarzanie


# utworzenie korpusu dokumentów
corpus_dir <- "./20_Dokumenty"
corpus <- VCorpus(
  DirSource(
    corpus_dir,
    "UTF-8",
    "*.txt"
  ),
  readerControl = list(
    language = "pl_PL"
  )
)

# dodatkowe funkcje transformujące
paste_paragraphs <- content_transformer(
  function(text){
    paste(text, collapse = " ")
  }
)
remove_char <- function(text, char) gsub(char, "", text)
cut_extension <- function(document){
  meta(document, "id") <- gsub("\\.txt$", "", meta(document, "id"))
  return(document)
}

# wstępne przetwarzanie
corpus <- tm_map(corpus, cut_extension)
corpus <- tm_map(corpus, paste_paragraphs)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, content_transformer(tolower))
stoplist_file <- "./stopwords_pl.txt"
stoplist <- readLines(
  stoplist_file,
  encoding = "utf-8"
)
corpus <- tm_map(corpus, removeWords, stoplist)
corpus <- tm_map(corpus, content_transformer(remove_char), intToUtf8(8722))
corpus <- tm_map(corpus, content_transformer(remove_char), intToUtf8(190))
corpus <- tm_map(corpus, content_transformer(trimws))
corpus <- tm_map(corpus, stripWhitespace)

# lematyzacja
polish <- dictionary("pl_PL")
lemmatize <- function(text){
  parsed_text_vec <- unlist(hunspell_parse(text, dict = polish))
  lemmatized_text_vec <- hunspell_stem(parsed_text_vec, dict = polish)
  for (t in 1:length(lemmatized_text_vec)) {
    if(length(lemmatized_text_vec[[t]]) == 0) lemmatized_text_vec[t] <- parsed_text_vec[t]
    if(length(lemmatized_text_vec[[t]])  > 1) lemmatized_text_vec[t] <- lemmatized_text_vec[[t]][1]
  }
  lemmatized_text <- paste(lemmatized_text_vec, collapse = " ")
  return(lemmatized_text)
}
corpus <- tm_map(corpus, content_transformer(lemmatize))

# eksport przetworzonego korpusu do plików
preprocessed_dir <- "./przetworzone"
dir.create(preprocessed_dir)
writeCorpus(corpus,preprocessed_dir)

## Tworzenie macierzy częstośći

corpus_dir <- "./przetworzone"
corpus <- VCorpus(
  DirSource(
    corpus_dir,
    "UTF-8",
    "*.txt"
  ),
  readerControl = list(
    language = "pl_PL"
  )
)

# dodatkowe funkcje transformujące
cut_extension <- function(document){
  meta(document, "id") <- gsub("\\.txt$", "", meta(document, "id"))
  return(document)
}

# wstępne przetwarzanie
corpus <- tm_map(corpus, cut_extension)

# tworzenie macierzy częstości
tdm_tf_all <- TermDocumentMatrix(corpus)

# transponowana tdm
dtm_tf_all <- DocumentTermMatrix(corpus)



tdm_tfidf_216 <- DocumentTermMatrix(
  corpus,
  control = list(
    weighting = weightTfIdf,
    bounds = list(
      global = c(2,16)
    )
  )
)


tdm_bin_410 <- DocumentTermMatrix(
  corpus,
  control = list(
    weighting = weightBin,
    bounds = list(
      global = c(4,10)
    )
  )
)


tdm_tf_612 <- DocumentTermMatrix(
  corpus,
  control = list(
    weighting = weightTf,
    bounds = list(
      global = c(6,12)
    )
  )
)


tdm_tfidf_216_m <- as.matrix(tdm_tfidf_216)
tdm_bin_410_m <- as.matrix(tdm_bin_410)
tdm_tf_612_m <- as.matrix(tdm_tf_612)

