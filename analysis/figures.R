library(here)
library(rio)
library(dplyr)
library(tidyr)
library(ggplot2)
library(glue)
library(emmeans)
library(forcats)

dir.create("figures")

theme_set(
  theme_bw() +
    theme(panel.spacing = unit(0.1, "lines"), legend.margin = margin(t = -5))
)

rename_data <- function(x) {
  if("dataset" %in% colnames(x))
    x$dataset <- fct_recode(x$dataset,
                            "CiteULike" = "citeulike",
                            "Epinions" = "Epinions",
                            "Sports & Outdoors" = "Home_and_Kitchen",
                            "Home & Kitchen" = "Sports_and_Outdoors"
    ) %>% fct_relevel("CiteULike", "Epinions", "Sports & Outdoors", "Home & Kitchen")
  if("metric_optim" %in% colnames(x))
    x$metric_optim <- fct_recode(tolower(x$metric_optim),
                                 "RR" = "rr",
                                 "AP" = "ap",
                                 "nDCG" = "dcg",
                                 "nDCG" = "ndcg",
                                 "nRBP.8" = "rbp_80",
                                 "nRBP.9" = "rbp_90",
                                 "nRBP.95" = "rbp_95",
                                 "nRBP.95" = "nrbp.95"
    ) %>% fct_relevel("RR", "AP", "nDCG", "nRBP.8", "nRBP.9", "nRBP.95")
  if("metric_eval" %in% colnames(x))
    x$metric_eval <- fct_recode(tolower(x$metric_eval),
                                "RR" = "rr",
                                "AP" = "ap",
                                "nDCG" = "ndcg",
                                "RBP.8" = "rbp_80",
                                "RBP.9" = "rbp_90",
                                "RBP.95" = "rbp_95",
                                "RBP.95" = "rbp.95"
    ) %>% fct_relevel("RR", "AP", "nDCG", "RBP.8", "RBP.9", "RBP.95")
  if("NSR" %in% colnames(x))
    x$NSR <- fct_recode(as.character(x$NSR),
                        "100%" = "1",
                        "200%" = "2",
                        "500%" = "5"
    )
  if("fold" %in% colnames(x))
    x$fold <- as.factor(x$fold)
  
  x
}

for(.FOLDER in c("results", "lambda_results")) {
  .FOLDER_TITLE <- ifelse(.FOLDER == "results", "Listwise", "Pairwise")
  
  d <- import(glue("analysis/{.FOLDER}_best.csv")) %>%
    rename_data()
  
  # Fig 1: overall performance #####################################################################
  
  pdf(glue("figures/fig_overall_{.FOLDER}.pdf"), width = 7, height = 6)
  print(
    ggplot(d, aes(metric_optim, y, color = NSR)) +
      geom_point() +
      geom_line(data = d %>%
                  group_by(dataset, NSR, metric_eval, metric_optim) %>%
                  summarize(y = mean(y)) %>%
                  mutate(metric_optim = as.integer(metric_optim))) +
      facet_grid(metric_eval ~ dataset, scales = "free_y") +
      labs(title = .FOLDER_TITLE, y = "Score") +
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1),
            legend.position = "bottom")
  )
  dev.off()
  
  # Fig 2: counts of best ##########################################################################
  
  d_best <- d %>%
    group_by(dataset, NSR, fold, metric_eval) %>%
    filter(y >= max(y)) %>%
    ungroup() %>%
    count(dataset, metric_optim, metric_eval, .drop = FALSE)
  
  pdf(glue("figures/fig_bestcount_{.FOLDER}.pdf"), width = 7, height = 2.7)
  print(
    ggplot(d_best, aes(metric_optim, metric_eval, fill = n)) +
      geom_tile() +
      scale_x_discrete(drop = FALSE) +
      facet_wrap(~ dataset, ncol = 4) +
      scale_fill_gradientn(colors = c("white", "black"), values = 0:1, name = "count") +
      labs(title = .FOLDER_TITLE) +
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
  )
  dev.off()
  
  # Fig 3-4: estimated marginal means ##############################################################
  
  e_d <- e <- NULL # to hold EMMs, faceted and not faceted
  for(emetric in levels(d$metric_eval)) {
    d2 <- d %>%
      filter(metric_eval == emetric) %>%
      # standardize scores within dataset and NSR
      group_by(dataset, NSR) %>%
      mutate(y_ = scale(y)[,1]) %>% # y_ is the scaled score
      ungroup()
    
    # fit model
    m <- lm(y_ ~ (metric_optim + NSR + dataset)^2 - NSR:dataset, d2)
    
    # compute EMM and intervals, faceted by dataset
    ee <- emmeans(m, ~ metric_optim | dataset)
    ci <- predict(ee, interval = "conf", adjust = "none", as.df = TRUE)
    pi <- predict(ee, interval = "pred", adjust = "none", as.df = TRUE)
    # put together and identify metric_eval
    ee <- inner_join(ci,pi, by = c("metric_optim", "dataset")) %>%
      mutate(metric_eval = emetric)
    # join with data from other metric_eval
    e_d <- bind_rows(e_d, ee)
    
    # compute EMM and intervals
    ee <- emmeans(m, ~ metric_optim)
    ci <- predict(ee, interval = "conf", adjust = "none", as.df = TRUE)
    pi <- predict(ee, interval = "pred", adjust = "none", as.df = TRUE)
    # put together and identify metric_eval
    ee <- inner_join(ci,pi, by = "metric_optim") %>%
      mutate(metric_eval = emetric)
    # join with data from other metric_eval
    e <- bind_rows(e, ee)
  }
  
  # plot unfaceted
  pdf(glue("figures/fig_effects_{.FOLDER}.pdf"), width = 3.5, height = 6)
  print(
    ggplot(e %>% filter(metric_optim != "RR"), aes(emmean, metric_optim, color = metric_optim)) +
      geom_point(size = 1.5) +
      geom_linerange(aes(xmin = lower.PL, xmax = upper.PL), size = .3) +
      geom_errorbar(aes(xmin = lower.CL, xmax = upper.CL), width = .5, size = .3) + 
      scale_color_manual(values = c("#619cff", "#f8766d", "#74c476", "#41ab5d", "#238b45")) +
      facet_wrap(~ metric_eval, ncol = 1) + 
      labs(title = .FOLDER_TITLE, x = "EMM Score (standardized)") +
      theme(legend.position = "none")
  )
  dev.off()
  
  # plot faceted
  pdf(glue("figures/fig_effects2_{.FOLDER}.pdf"), width = 7, height = 6)
  print(
    ggplot(e_d %>% filter(metric_optim != "RR"), aes(emmean, metric_optim, color = metric_optim)) +
      geom_point(size = 1.5) +
      geom_linerange(aes(xmin = lower.PL, xmax = upper.PL), size = .3) +
      geom_errorbar(aes(xmin = lower.CL, xmax = upper.CL), width = .5, size = .3) + 
      facet_grid(metric_eval ~ dataset) + 
      scale_color_manual(values = c("#619cff", "#f8766d", "#74c476", "#41ab5d", "#238b45")) +
      labs(title = .FOLDER_TITLE, x = "EMM Score (standardized)") +
      theme(legend.position = "none")
  )
  dev.off()
  
  # Fig 5: score differences #######################################################################
  
  d_ind <- NULL
  for(dname in unique(d$dataset)) {
    # read number of interactions per user
    dname2 <- gsub(" & ","_and_",dname, fixed = TRUE)
    ni <- import(glue("analysis/user_freq_{dname2}.csv"))
    
    # find out the best learning rates for each metric_optim
    b <- d %>%
      filter(fold == 0, NSR == "500%", metric_optim %in% c("AP", "nDCG","nRBP.95"))
    
    for(ometric in unique(b$metric_optim)) {
      b2 <- b %>% filter(metric_optim == ometric, dataset == dname)
      
      # for each metric_eval, read all individual results and aggregate
      for(emetric in c("AP", "nDCG","RBP.95")) {
        lr <- b2 %>% filter(metric_eval == emetric) %>% pull(lr)
        lr <- ifelse(lr < 1, lr, glue("{lr}.0"))
        th <- ifelse(dname %in% c("CiteULike", "Epinions"), 0, 4)
        m <- list(RR = "rr", AP = "ap", nDCG = "dcg",
                  nRBP.8 = "rbp_80", nRBP.9 = "rbp_90", nRBP.95 = "rbp_95")[[ometric]]
        
        for(fold in 0:2) {
          if(.FOLDER == "results") {
            dd <- import(glue("{.FOLDER}/{dname2}/individual/loss_type_{m}_",
                              "lr_{lr}_th_{th}_reg_0_fold_{fold}_frac_5.0_emb_size_32.csv"))
          } else {
            ometric2 <- gsub("rbp_95", "rbp_0.95", m)
            dd <- import(glue("{.FOLDER}/{dname2}/individual/loss_type_lambda_{ometric2}_",
                              "lr_{lr}_th_{th}_reg_0_fold_{fold}_frac_5.0_emb_size_32.csv"))
          }
          
          dd <- dd %>%
            select(all_of(list(AP = "AP", nDCG = "NDCG", RBP.95 = "RBP_95")[[emetric]]))
          dd <- data.frame(y = dd[,1],
                           NSR = "500%", dataset = dname, fold = fold,
                           metric_optim = ometric, metric_eval = emetric,
                           ni = ni$freq, user_id = ni$user_id)
          d_ind <- bind_rows(d_ind, dd)
        }
      }
    }
  }
  d_ind <- rename_data(d_ind)
  
  # join with itself, so that we have results for AP/nDCG and RBP.95 next to each other
  dd <- left_join(d_ind, d_ind %>% filter(metric_optim == "nRBP.95"),
                  by = c("dataset", "NSR", "fold", "metric_eval", "user_id")) %>%
    filter(metric_optim.x != "nRBP.95") %>%
    rename(metric_optim = metric_optim.x)
  
  pdf(glue("figures/fig_ni_{.FOLDER}.pdf"), width = 7, height = 5)
  print(
    ggplot(dd, aes(ni.x, y.y - y.x, color = metric_optim, fill = metric_optim)) +
      geom_smooth(se = TRUE, formula = y ~ s(x, k = 4)) +
      geom_hline(yintercept = 0, linetype = 2) +
      scale_x_log10() +
      facet_grid(metric_eval ~ dataset, scales = "free") +
      labs(title = glue("{.FOLDER_TITLE}, NSR=500%"),
           x = "Number of positive items", y = "nRBP.95 - metric_optim") +
      theme(legend.position = "bottom")
  )
  dev.off()
}
