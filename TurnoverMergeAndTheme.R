# Adapted R code provided by 
# Speer, A. B., Perrotta, J., Tenbrink, A. P., Wegmeyer, L. J., Delacruz, A. Y., & Bowker, J. (2023). Turning words into numbers: Assessing work attitudes using natural language processing. Journal of Applied Psychology, 108(6), 1027.

# Clear Environment
rm(list = ls())

# Load Necessary Libraries (Removed duplicates and unused ones)
library(dplyr)
library(stringi)
library(data.table)
library(tm)
library(textstem)
library(RWeka)
library(quanteda)
library(readr)
library(tidyr)

# File Selection
file_csv <- choose.files()

# Function: Clean Text
do_clean_text <- function(x) {
  x <- stri_replace_all_regex(x, "\"[^\"]*\"", "")
  x <- tolower(x)
  x <- qdap::replace_contraction(x) %>% 
    qdap::replace_abbreviation() %>%
    qdap::replace_ordinal() %>%
    qdap::replace_number() %>%
    qdap::replace_symbol()
  x <- strip(x) %>% textstem::lemmatize_strings() %>% tm::stripWhitespace()
  return(x)
}

# Load Theme Dictionary (See Speer et al. 2023 to get the dictionary file)
dict <- read_csv("TAPS Theme Dictionary SIMPLE.csv", col_types = cols(.default = "c")) %>%
  mutate(Phrase = do_clean_text(Phrase)) %>%
  distinct()
theme_constructs <- unique(dict$Construct)

# Tokenizer Function
tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 2)) 

# Function: Append Theme Score
append_theme_score <- function(df) {
  df <- df %>% mutate(ID = row_number(), Word_Count = str_count(Text, "\\w+"))
  
  Text_df <- df %>% filter(!is.na(Text)) %>%
    select(ID, Text) %>%
    rename(doc_id = ID, text = Text) %>%
    mutate(text = do_clean_text(text))
  
  # Convert to Corpus & Create DTM
  corpus <- VCorpus(DataframeSource(Text_df))
  dtm <- DocumentTermMatrix(corpus, control = list(tokenize = tokenizer)) %>%
    removeSparseTerms(0.9995) %>% # Reduce dataset size
    as.matrix() %>%
    as.data.frame()
  
  # Initialize Theme Score Output
  theme_output <- data.frame(ID = Text_df$doc_id)
  
  # Vectorized Theme Score Calculation
  for (construct in theme_constructs) {
    dict_temp <- filter(dict, Construct == construct)
    term_cols <- intersect(colnames(dtm), dict_temp$Phrase)
    
    if (length(term_cols) > 0) {
      theme_count <- rowSums(dtm[, term_cols, drop = FALSE])
      theme_score <- (theme_count / (df$Word_Count + 1e-40)) * 100
      
      theme_output <- theme_output %>%
        mutate(!!paste0("Theme_Count_", construct) := theme_count,
               !!paste0("Theme_Score_", construct) := theme_score)
    }
  }
  
  # Merge Theme Scores
  df <- left_join(df, theme_output, by = "ID")
  return(df)
}

# Function: Drop Deleted Rows
drop_deleted <- function(df) {
  df <- df[!apply(df == "[deleted]", 1, any), ]
  return(df)
}

# Process CSV Files
df <- data.frame()
for (file in file_csv) {
  print(file)
  
  df_next <- read_csv(file, col_types = cols(.default = "c"))
  source <- tools::file_path_sans_ext(basename(file))
  
  # Identify Data Source Type
  if (grepl("_threads", source, fixed = TRUE)) {
    df_next <- df_next %>%
      mutate(source = gsub("_threads", "", source),
             is_twitter = FALSE, is_thread = TRUE,
             Text = paste(title, text)) %>%
      select(-title, -text)
  } else if (grepl("_comments", source, fixed = TRUE)) {
    df_next <- df_next %>%
      mutate(source = gsub("_comments", "", source),
             is_twitter = FALSE, is_thread = FALSE,
             Text = comment) %>%
      select(-comment)
  } else {
    setnames(df_next, "tweet", "Text")
    df_next <- df_next %>% mutate(is_twitter = TRUE)
  }
  
  df_next <- drop_deleted(df_next) %>% append_theme_score()
  
  # Save Temp Output
  write_csv(df_next, "temp_output.csv", na = "")
  
  # Use `bind_rows()` Instead of `rbind.fill()` for Performance
  df <- bind_rows(df, df_next)
}

# Assign Unique IDs
df <- df %>% mutate(ID = row_number()) %>% filter(Text != "[deleted]")

# Save as RDS and CSV (Ensure Encoding)
saveRDS(df, file = "df.rds")
df[] <- lapply(df, function(x) if (is.character(x)) iconv(x, from = "", to = "UTF-8", sub = "byte") else x)
write_csv(df, "df.csv", na = "")

# Save as RDS and CSV (Dropping Theme=0)
df_filtered <- df %>%
  group_by(url) %>%
  mutate(mean_theme_score = mean(Theme_Score_Turnover_Intentions, na.rm = TRUE)) %>%
  ungroup() %>%
  filter(mean_theme_score != 0) %>%
  select(-mean_theme_score)  # Remove the temporary mean column

saveRDS(df, file = "df_short.rds")
write_csv(df, "df_short.csv", na = "")