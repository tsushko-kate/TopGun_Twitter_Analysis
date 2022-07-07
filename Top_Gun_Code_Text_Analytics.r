
###-----------------------------------!
### Install packages (ONLY ONCE) -----
###-----------------------------------!
install.packages("SentimentAnalysis")
install.packages("vader")
install.packages("syuzhet")
install.packages("topicmodels")
install.packages("quanteda")
install.packages("wordcloud")
install.packages("BTM")
install.packages("udpipe")
install.packages("data.table")
install.packages("stopwords")
install.packages("textplot")
install.packages("ggraph")
install.packages("stringr")
install.packages("concaveman")
install.packages("tidytext")
install.packages("quanteda.textplots")


##-------------------------!
##  Read the data file ----
##-------------------------!
file_name = "starbucks.hashtag" ## Specify This
d <- read.csv(paste(file_name,".csv",sep=""),stringsAsFactors = F)



##------------------------!
##  Prepare the texts ----
##------------------------!

unclean_text <- tolower(d$text)

clean_text = gsub("&amp", "", unclean_text)
clean_text = gsub("@\\w+", "", clean_text)
clean_text = gsub("[[:punct:]]", "", clean_text)
clean_text = gsub("[[:digit:]]", "", clean_text)
clean_text = gsub("http\\w+", "", clean_text)
clean_text = gsub("[ \t]{2,}", "", clean_text)
clean_text = gsub("^\\s+|\\s+$", "", clean_text) 
clean_text = gsub("Ã¢","",clean_text)
clean_text = gsub("???","",clean_text)
clean_text = gsub("T","",clean_text)

d$text_cleaned = clean_text


##-------------------------------------------------------!
## Sentiment Analysis using analyzeSentiment package ----
##-------------------------------------------------------!
library(SentimentAnalysis)
sentiment1 = analyzeSentiment(d$text_cleaned)
d1 = cbind(d,sentiment1[,c("SentimentGI","SentimentHE","SentimentLM","SentimentQDAP")])

##---------------------------------------------!
##  Sentiment Analysis using vader package ----
##---------------------------------------------!
library(vader)
sentiment2 <- vader_df(d$text_cleaned)
names(sentiment2)[3] = "compound.valence"
d2 = cbind(d1,sentiment2[,c("compound.valence")])
names(d2)[ncol(d2)] = "compound.valence"


##-----------------------------------------------!
##  Emotion Analysis using "syuzhet" package ----
##-----------------------------------------------!
library(syuzhet)
emotions = get_nrc_sentiment(d$text_cleaned)
d3 = cbind(d2,emotions[,c(1:8)])



##-------------------------------------------------!
##  Pre-processing the text for topic modeling ----
##-------------------------------------------------!
require(quanteda)
tokensAll = tokens(d$text_cleaned, remove_punct = TRUE)
tokensNoStopwords = tokens_remove(tokensAll, c(stopwords("english"),"T","???","starbucks","get","???","s","t","uf","???","T","???_T","im","â"))
tokensNgramsNoStopwords = tokens_ngrams(tokensNoStopwords, c(1,2))
myDFM = dfm(tokensNgramsNoStopwords)
myDFM = dfm_trim(myDFM ,  min_termfreq = 2, min_docfreq = 2)
topfeatures(myDFM, 20)


##----------------------------------------------!
##  Look into the terms in their context -------
##----------------------------------------------!
term = "fashion" ## Specify This
corpus(d$text)[which(as.numeric(myDFM[,term])>0)]


##--------------------------!
##  Create a word cloud ----
##--------------------------!
require(quanteda.textplots)
textplot_wordcloud(myDFM,
                   min_count = 20,
                   random_order = T,
                   rotation = .25,
                   color = RColorBrewer::brewer.pal(10,"Dark2"))


##-------------------------------!
##  Topic modeling using LDA ----
##-------------------------------!

require(topicmodels)
f <- convert(myDFM, to="topicmodels")

# Determining Optimal Number of Topics
# library("ldatuning")
# result <- FindTopicsNumber(
#   myDFM,
#   topics = seq(from = 5, to = 10, by = 1),
#   metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
#   method = "Gibbs",
#   control = list(seed = 78),
#   mc.cores = 2L,
#   verbose = TRUE
# )
# FindTopicsNumber_plot(result)


K <- 7 ## Number of topics
lda.model <- LDA(f, k = K, method = "Gibbs", control = list(verbose=25L, seed = 123, burnin = 100, iter = 500))
terms(lda.model,10) ## top 10 terms in each topic

require(dplyr)
require(tidytext)
require(ggplot2)


topic_lda <- tidy(lda.model,matrix = "beta")
top_terms <- topic_lda %>%
  group_by(topic) %>%
  top_n(5,beta) %>% 
  ungroup() %>%
  arrange(topic,-beta)
plot_topic <- top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  scale_x_reordered()
plot_topic


write.csv(terms(lda.model,20),"topics.and.terms.lda.csv") ## top 20 terms in each topic

## Extracting Topics of each text --> Combine with the Original Data
topics = as.data.frame(topics(lda.model))
names(topics)[1] = "topic_lda"
topics$text_id = row.names(topics)
row.names(topics) = NULL
topics = topics[,c(2,1)]
d3$text_id = paste("text",seq(1,nrow(d3),1),sep="")
d4 = merge(d3,topics,by="text_id",sort=F,all=T)


##----------------------------------!
##  Topic modeling using Biterm ----
##----------------------------------!
library(BTM)
library(udpipe)
library(data.table)
library(stopwords)
library(textplot)
library(ggraph)
library(stringr) 

anno <- udpipe(d$text, "english", trace = 100)
biterms <- as.data.table(anno)
traindata <- subset(anno, upos %in% c("NOUN", "ADJ", "VERB") & !lemma %in% stopwords("en") &
                      nchar(lemma) > 2 & !str_detect(token,"@+") & !str_detect(token,"#+") & !str_detect(token,"starbucks"))
traindata <- traindata[, c("doc_id", "lemma")]
K <- 10 ## Number of topics
btm.model <- BTM(traindata, k = K, alpha = 1, beta = 0.01, iter = 100, trace = TRUE,window=100,detailed = T)
terms(btm.model,top_n = 20)
write.csv(terms(btm.model,top_n = 10),"topics.and.terms.biterm.csv") ## top 10 terms in each topic
plot(btm.model,top_n = 10)
scores <- as.data.frame(predict(btm.model, newdata = traindata,sort=F))
scores$topic_btm = apply(scores,1,which.max)
scores$text_id = row.names(scores)
scores$text_id = gsub("doc","text",scores$text_id)
row.names(scores) = NULL
d5 = merge(d4,scores[,c("text_id","topic_btm")],by="text_id",sort=F,all=T)
d5$created_at_date = as.character(as.Date(d5$created_at))


##---------------------------------------------!
##  Saving the data file on the hard drive ----
##---------------------------------------------!
write.csv(d5,paste(file_name,"_with_sentiment_and_topics.csv",sep=""))




