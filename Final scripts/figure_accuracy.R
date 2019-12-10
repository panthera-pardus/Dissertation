library(tidyverse)
library(ggplot2)

setwd("/Users/andour/Google Drive/projects/Dissertation/Final data")
F1_df <- read_csv("F1_df.csv") %>% 
  select(-X1, -label) %>% 
  reshape2::melt(data = ., id.vars = c("noise_bucket", "drift")) %>% 
  mutate(drift = if_else(drift == T, "With drift", "Without drift"))


ggplot(F1_df, aes(x = noise_bucket, y = value, color = variable)) +
  geom_line(size = 1) +
  xlab("Noise level") +
  ylab("F1 score") +
  xlim(0.1,1) +
  scale_color_manual(values=c("#FF6600","#2164F3", "#FFB100", "#008040", "#CD29C0", "#551A8B", "#0000CC", "#99CCFF", "#666666", "#000000"), 
                    name = "Model selection strategy ", 
                    labels = c("BF", "WAIC", "MSE", "MAE", "R^2", "Chi^2", "AIC", "BIC", "Entropy BIC", "Entropy AIC")) +
  facet_grid(drift~.) +
  theme(
    panel.background = element_rect(fill = "#E8EEF7", colour = "#E8EEF7",
                                    size = 2, linetype = "solid"),
    panel.grid.major = element_line(size = 0.5, linetype = 'solid',
                                    colour = "white"), 
    panel.grid.minor = element_line(size = 0.25, linetype = 'solid',
                                    colour = "white")
  )
  ggsave("/Users/andour/Google Drive/projects/Dissertation/Final figures/F1_noise.png")

  