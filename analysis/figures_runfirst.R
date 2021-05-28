library(here)
library(rio)
library(dplyr)
library(tidyr)
library(ggplot2)
library(glue)
library(emmeans)

for(.FOLDER in c("results", "lambda_results")) {
  # read data
  d <- lapply(list.dirs(.FOLDER, recursive = FALSE, full.names = FALSE), function(dname) {
    import(glue("{.FOLDER}/{dname}/all_res.csv")) %>%
      mutate(dataset = dname)
  }) %>%
    bind_rows() %>%
    filter(emb_size == 32) %>%
    select(-threshold, -reg, -emb_size) %>%
    mutate(loss_type = gsub("lambda_", "", loss_type)) %>%
    filter(loss_type %in% c("dcg", "ap", "rr", "rbp_80", "rbp_90", "rbp_95")) %>%
    mutate(loss_type = factor(loss_type)) %>%
    rename(metric_optim = loss_type, NSR = frac) %>% 
    group_by(lr) %>% filter(n()>100) %>% ungroup()
  
  # turn into long format
  d <- d %>%
    pivot_longer(cols = `NDCG@5`:RBP_95, names_to = "metric_eval", values_to = "y") %>%
    filter(metric_eval %in% c("AP", "NDCG", "RR", "RBP_80", "RBP_90", "RBP_95"))
  
  # compute best LR and filter out the others
  d <- d %>%
    group_by(dataset, fold, NSR, metric_optim, metric_eval) %>%
    filter(y >= max(y)) %>%
    slice_head(n = 1) %>%
    ungroup()
  
  export(d, glue("analysis/{.FOLDER}_best.csv"))
}
