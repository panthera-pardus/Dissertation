library(tidyverse)
library(ggplot2)

setwd("/Users/andour/Google Drive/projects/Dissertation/Final data")

# Linear parameter --------------------------------------------------------

linear_class_data <- read_csv("shanon_bic_waic_linear.csv") %>% 
  select(noise_bucket, param_estimation,`Estimation Percentage Freq`, `Estimation Percentage Bayes`) %>% 
  reshape2::melt(., id.vars = c("noise_bucket", "param_estimation"))

ggplot(freq_class_data, aes(noise_bucket, value, fill=param_estimation)) + 
  geom_bar(stat="identity") +
  xlab("Noise level") +
  ylab("Percentage (%) of estimated parameters") +
  scale_fill_manual(values=c("#FF6600","#2164F3"), name = "Share of parameters : ", labels = c("Wrongly estimated", "Correctly estimated"))+
  ggtitle("Linear parameter estimations in entropy enhanced BIC and WAIC true positives") +
  facet_wrap(~variable, labeller = labeller(variable = 
                                              c("Estimation Percentage Freq" = "Entropy BIC",
                                                "Estimation Percentage Bayes" = "WAIC")
  )) + 
  theme(
    panel.background = element_rect(fill = "#E8EEF7", colour = "#E8EEF7",
                                    size = 2, linetype = "solid"),
    panel.grid.major = element_line(size = 0.5, linetype = 'solid',
                                    colour = "white"), 
    panel.grid.minor = element_line(size = 0.25, linetype = 'solid',
                                    colour = "white")
  ) +
  ggsave("/Users/andour/Google Drive/projects/Dissertation/Final figures/linear_param_best.png")



# Logistic Parameters -----------------------------------------------------

K_logistic_class_data <- read_csv("k_shanon_bic_waic_logistic.csv") %>% 
  select(noise_bucket, param_estimation,`Estimation Percentage Freq`, `Estimation Percentage Bayes`) %>% 
  reshape2::melt(., id.vars = c("noise_bucket", "param_estimation")) %>% 
  rename("k" = "value", "Model selection" = "variable") %>% 
  mutate(`Model selection` = if_else(`Model selection` == "Estimation Percentage Freq", "Entropy BIC", "WAIC"))

x0_logistic_class_data <- read_csv("x0_shanon_bic_waic_logistic.csv") %>% 
  select(noise_bucket, param_estimation,`Estimation Percentage Freq`, `Estimation Percentage Bayes`) %>% 
  reshape2::melt(., id.vars = c("noise_bucket", "param_estimation")) %>% 
  rename("x0" = "value", "Model selection" = "variable") %>% 
  mutate(`Model selection` = if_else(`Model selection` == "Estimation Percentage Freq", "Entropy BIC", "WAIC"))

logistic_class_data <- left_join(K_logistic_class_data, x0_logistic_class_data) %>% 
  reshape2::melt(., id.vars = c("noise_bucket", "param_estimation", "Model selection")) 


ggplot(logistic_class_data, aes(noise_bucket, value, fill=param_estimation)) + 
  geom_bar(stat="identity") +
  xlab("Noise level") +
  ylab("Percentage (%) of estimated parameters") +
  scale_fill_manual(values=c("#FF6600","#2164F3"), name = "Share of parameters : ", labels = c("Wrongly estimated", "Correctly estimated"))+
  ggtitle("Logistic parameter estimations in entropy enhanced BIC and WAIC true positives") +
  facet_grid(variable~`Model selection`, labeller = label_both) + 
  theme(
    panel.background = element_rect(fill = "#E8EEF7", colour = "#E8EEF7",
                                    size = 2, linetype = "solid"),
    panel.grid.major = element_line(size = 0.5, linetype = 'solid',
                                    colour = "white"), 
    panel.grid.minor = element_line(size = 0.25, linetype = 'solid',
                                    colour = "white")
  ) +
  ggsave("/Users/andour/Google Drive/projects/Dissertation/Final figures/logistic_param_best.png")

