# załadowanie bibliotek
library(tm)
library(hunspell)
library(topicmodels)
library(wordcloud)
library(lsa)
library(proxy)
library(dendextend)
library(corrplot)
library(flexclust)


### Tworzenie korpusu i wstępne przetwarzanie ###


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


### Tworzenie macierzy częstości ###

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

dtm_tf_all <- DocumentTermMatrix(corpus)

dtm_tfidf_all <- DocumentTermMatrix(
  corpus,
  control = list(
    weighting = weightTfIdf
  )
)

tdm_tf_612 <- TermDocumentMatrix(
  corpus,
  control = list(
    weighting = weightTf,
    bounds = list(
      global = c(6,12)
    )
  )
)

tdm_tfidf_410 <- TermDocumentMatrix(
  corpus,
  control = list(
    weighting =  weightTfIdf,
    bounds = list(
      global = c(4,10)
    )
  )
)

dtm_tf_612 <- DocumentTermMatrix(
  corpus,
  control = list(
    weighting = weightTf,
    bounds = list(
      global = c(6,12)
    )
  )
)


dtm_tfidf_410 <- DocumentTermMatrix(
  corpus,
  control = list(
    weighting = weightTfIdf,
    bounds = list(
      global = c(4,10)
    )
  )
)

tdm_tf_all_m <- as.matrix(tdm_tf_all)
tdm_tf_612_m <- as.matrix(tdm_tf_612)
tdm_tfidf_410_m <- as.matrix(tdm_tfidf_410)
dtm_tf_all_m <- as.matrix(dtm_tf_all)
dtm_tfidf_all_m <- as.matrix(dtm_tfidf_all)
dtm_tf_612_m <- as.matrix(dtm_tf_612)
dtm_tfidf_410_m <- as.matrix(dtm_tfidf_410)


### Redukcja wymiarów macierzy ###

# utworzenie katalogu na wyniki
reduction_dir <- "./reduction"
dir.create(reduction_dir)

# legenda
#matrix
doc_names <- rownames(dtm_tf_all)
doc_count <- length(doc_names)
legend <- paste(
  paste(
    "d",
    1:doc_count,
    sep = ""
  ),
  doc_names,
  sep = " -> "
)
options(scipen = 5)

# kolory
clusters_pattern <- c(1,1,1,1,2,2,2,2,4,4,4,4,5,5,5,5,6,6,6,6)
cols = c("lightsteelblue", "orchid", "royalblue","darkseagreen1", "cyan4", "navy" , "mediumpurple1")
cols_pattern <- cols[clusters_pattern]
names(clusters_pattern) <- doc_names
names(cols_pattern) <- doc_names

#experiment matrix
#exp_matrix <- dtm_tf_all
#exp_matrix <- dtm_tf_612
exp_matrix <- dtm_tfidf_410


#Analiza głównych składowych
pca_model <- prcomp(exp_matrix)

x <- pca_model$x[,1]
y <- pca_model$x[,2]

plot_file <- paste(
  reduction_dir,
  "pca.png",
  sep = "/"
)
png(plot_file, width = 800)
par(mar = c(4, 4, 4, 25), xpd = TRUE)
plot(
  x,
  y,
  main = "Analiza głównych składowych",
  xlab = "PC1",
  ylab = "PC2",
  col = cols_pattern,
  pch = 16
)
text(
  x,
  y,
  paste(
    "d",
    1:doc_count,
    sep = ""
  ),
  col = cols_pattern,
  pos = 1
)
legend(
  "topright",
  inset = c(-0.9, 0.1),
  legend,
  text.col = cols_pattern
)
dev.off()

### Dekompozycja według wartości osobliwych ###

#experiment matrix
exp_matrix <- tdm_tf_all
#exp_matrix <- tdm_tf_612
#exp_matrix <- tdm_tfidf_410

# analiza ukrytych wymiarów semantycznych
lsa_model <- lsa(exp_matrix)

coord_docs <- lsa_model$dk%*%diag(lsa_model$sk)
coord_terms <- lsa_model$tk%*%diag(lsa_model$sk)

terms_importance <- diag(
  lsa_model$tk%*%diag(lsa_model$sk)%*%t(diag(lsa_model$sk))%*%t(lsa_model$tk)
)
important_terms <- names(
  tail(
    sort(terms_importance),
    30
  )
)

own_terms <- c("daenerys", "jon", "cersei", "eragon", "bilbo", "yennefer", "ciri", "jaskier", "geralt", "frodo", "gandalf", "halt", "ork", "horace")
#own_terms <- c("elf", "czarodziej", "smok", "brat", "arya", "cel", "dawny")
#own_terms <- c("catelyn", "brama", "bohater", "bagginsa", "bronić","bezpieczny", "atakować")
current_terms <- own_terms

x1 <- coord_docs[,1]
y1 <- coord_docs[,2]
x2 <- coord_terms[current_terms,1]
y2 <- coord_terms[current_terms,2]

plot_file <- paste(
  reduction_dir,
  "lsa.png",
  sep = "/"
)
png(plot_file, width = 800)
par(mar = c(4, 4, 4, 25), xpd = TRUE)
plot(
  x1,
  y1,
  main = "Analiza ukrytych wymiarów semantycznych",
  xlab = "SD1",
  ylab = "SD2",
  col = cols_pattern,
  pch = 16
)
text(
  x1,
  y1,
  paste(
    "d",
    1:doc_count,
    sep = ""
  ),
  col = cols_pattern,
  pos = 1
)
points(
  x2,
  y2,
  pch = 15,
  col = "mediumvioletred"
)
text(
  x2,
  y2,
  rownames(coord_terms[current_terms,]),
  col = "mediumvioletred",
  pos = 2
)
legend(
  "topright",
  inset = c(-0.9, 0.1),
  legend,
  text.col = cols_pattern
)
dev.off()

### Analiza skupień ###

# utworzenie katalogu na wyniki
clusters_dir <- "./clusters"
dir.create(clusters_dir)

# analiza skupień

clusters_pattern <- c(1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5)
cols_pattern <- cols[clusters_pattern]

frequency_matrix <- dtm_tfidf_all_m
measure <- "cosine"
method <- "ward.D2"
name <- "tfidf_all"

doc_names <- rownames(frequency_matrix)
doc_count <- length(doc_names)
names(clusters_pattern) <- doc_names
names(cols_pattern) <- doc_names

dist_matrix <- dist(frequency_matrix, method = measure)
hierarch_clust <- hclust(dist_matrix, method = method)
plot_file <- paste(
  clusters_dir,
  paste("dend_",name,"_base.png", sep = ""),
  sep = "/"
)
png(plot_file)
plot(hierarch_clust)
dev.off()
dend_h <- as.dendrogram(hierarch_clust)
clusters_count <- find_k(dend_h)$k
plot_file <- paste(
  clusters_dir,
  paste("dend_",name,"_color.png", sep = ""),
  sep = "/"
)
dend_colored <- color_branches(
  dend_h,
  k = clusters_count,
  col = cols
)
png(plot_file, width = 600)
par(mai = c(1,1,1,4))
plot(dend_colored, horiz = T)
dev.off()
plot_file <- paste(
  clusters_dir,
  paste("dend_",name,"_pattern.png", sep = ""),
  sep = "/"
)
dend_colored <- color_branches(
  dend,
  col = cols_pattern[dend %>% labels]
)
png(plot_file, width = 600)
par(mai = c(1,1,1,4))
plot(dend_colored, horiz = T)
dev.off()
clusters <- cutree(hierarch_clust, k = clusters_count)
clusters_matrix = matrix(0, doc_count, clusters_count)
rownames(clusters_matrix) <- doc_names
for (doc in 1:doc_count) {
  clusters_matrix[doc, clusters[doc]] <- 1
}
plot_file <- paste(
  clusters_dir,
  paste("matrix_",name,".png", sep = ""),
  sep = "/"
)
png(plot_file)
# par(mai = c(1,1,1,4))
corrplot(clusters_matrix)
dev.off()

rand_pattern_experiment <- comPart(clusters, clusters_pattern)
#rand_experiment1_experiment2 <- comPart(clusters_1, clusters_2)

plot_file <- paste(
  clusters_dir,
  "FM_index.png",
  sep = "/"
)
png(plot_file)
Bk_plot(
  dend_h,
  dend_h,
  add_E = F,
  rejection_line_asymptotic = F,
  main = "Indeks Fawlkes'a - Mallows'a"
)
dev.off()

### LDA ###

# utworzenie katalogu na wyniki
topics_dir <- "./topics"
dir.create(topics_dir)

# Metoda Ukrytej Alokacji Dirichlet'a (Latent Dirichlet Allocation method)

#matrix <- dtm_tf_all
matrix <- dtm_tf_612
#topics_count <- 3
#topics_count <- 5
topics_count <- 7
lda_model <- LDA(
  matrix,
  topics_count,
  method = "Gibbs",
  control = list(
    burnin = 2000,
    thin = 100,
    iter = 3000
  )
)
results <- posterior(lda_model)

# kolory
cols = c("lightsteelblue", "orchid", "royalblue","darkseagreen1", "cyan4", "navy", "mediumpurple1")

# prezentacja tematów
for (topic_no in 1:topics_count) {
  topic_file <- paste(
    topics_dir,
    paste("Temat", topic_no, ".png"),
    sep = "/"
  )
  png(topic_file)
  par(mai = c(1,2,1,1))
  topic <- tail(sort(results$terms[topic_no,]),20)
  barplot(
    topic,
    horiz = TRUE,
    las = 1,
    main = paste("Temat", topic_no),
    xlab = "Prawdopodobieństwo",
    col = cols[topic_no]
  )
  dev.off()
}

#prezentacja dokumentów
plot_file <- paste(topics_dir, "Dokumenty.png",sep = "/")
png(plot_file)
par(mai = c(1,4,1,1))
barplot(
  t(results$topics),
  horiz = TRUE,
  las = 1,
  main = "Dokumenty",
  xlab = "Prawdopodobieństwo",
  col = cols
)
dev.off()


### Słowa kluczowe ###

# utworzenie katalogu na wyniki
clouds_dir <- "./clouds"
dir.create(clouds_dir)

# waga tf jako miara ważności słów w dokumentach
for (doc_no in 1:length(corpus)) {
  print(rownames(dtm_tf_all_m)[doc_no])
  print(head(sort(dtm_tf_all_m[doc_no,], decreasing = T)))
}

# waga tfidf jako miara ważności słów w dokumentach
for (doc_no in 1:length(corpus)) {
  print(rownames(dtm_tfidf_all_m)[doc_no])
  print(head(sort(dtm_tfidf_all_m[doc_no,], decreasing = T)))
}

# waga tf_612 jako miara ważności słów w dokumentach
for (doc_no in 1:length(corpus)) {
  print(rownames(dtm_tf_612_m)[doc_no])
  print(head(sort(dtm_tf_612_m[doc_no,], decreasing = T)))
}

# waga tf_612 jako miara ważności słów w dokumentach
for (doc_no in 1:length(corpus)) {
  print(rownames(tdm_tf_612_m)[doc_no])
  print(head(sort(tdm_tf_612_m[doc_no,], decreasing = T)))
}

# waga tf_410 jako miara ważności słów w dokumentach
for (doc_no in 1:length(corpus)) {
  print(rownames(tdm_tfidf_410_m)[doc_no])
  print(head(sort(tdm_tfidf_410_m[doc_no,], decreasing = T)))
}


#chmury tagów
for (doc_no in 1:length(corpus)) {
  cloud_file <- paste(
    clouds_dir,
    paste(corpus[[doc_no]]$meta$id, ".png", sep = ""),
    sep = "/"
  )
  png(cloud_file)
  wordcloud(
    corpus[doc_no],
    max.words = 200,
    colors = brewer.pal(8,"Blues")
  )
  dev.off()
}
