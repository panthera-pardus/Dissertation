library(reshape2)
library(ggplot2)


task1 <- c('Project Plan', '2019-04-01', '2019-04-30',1)
task2 <- c('Familiarise with\nPython Bayesian ecosystem', '2019-05-01', '2019-05-15',1)
task3 <- c('Generate data', '2019-05-15', '2019-05-31',1)
task4 <- c('Generate noise filters', '2019-05-15', '2019-05-31',1)

task5 <- c('Experiment 1 :\nSimple Evaluation', '2019-06-01', '2019-07-15',1)
task51 <- c('Test model 1','2019-06-01', '2019-06-15',2)
task52 <- c('Probability estimations 1','2019-06-15', '2019-07-01',2)
task53 <- c('Analyse results and\nwrite summary 1', '2019-07-01',  '2019-07-15',2)



task6 <- c('Experiment 2 :\nSystematic error', '2019-07-15', '2019-08-15',1)
task61 <- c('Test model 2','2019-07-15', '2019-08-01',2)
task62 <- c('Probability estimations 2','2019-07-15', '2019-08-01',2)
task63 <- c('Analyse results and\nwrite summary 2', '2019-08-01',  '2019-08-15',2)

task7 <- c('Experiment 3 :\nReal dataset', '2019-08-15', '2019-10-01',1)
task71 <- c('Preprocess data 3','2019-08-15', '2019-09-01',2)
task72 <- c('Test model 3','2019-09-01', '2019-09-15',2)
task73 <- c('Probability estimations 3','2019-09-15', '2019-10-01',2)
task74 <- c('Analyse results and\nwrite summary 3', '2019-09-15', '2019-10-01',2)


task8 <- c('Experiment 4 :\nReal dataset', '2019-10-01', '2019-11-01',1)
task81 <- c('Preprocess data 4','2019-10-01', '2019-10-10',2)
task82 <- c('Test model 4','2019-10-10', '2019-10-20',2)
task83 <- c('Probability estimations 4','2019-10-20', '2019-11-01',2)
task84 <- c('Analyse results and\nwrite summary 4', '2019-10-20', '2019-11-01',2)


task9 <- c('Writeup', '2019-07-01', '2019-12-01',1)



df <- as.data.frame(rbind(task1, task2, task3, task4, task5, task51, task52, task53, task6,task61,
                          task62,task63, task7,task71,task72,task73,task74,
                          task8, task81, task82, task83, task84, task9))
names(df) <- c('task', 'start', 'end', 'task_level')
df$task <- factor(df$task, levels = df$task)
df$start <- as.Date(df$start)
df$end <- as.Date(df$end)
df_melted <- melt(df, measure.vars = c('start', 'end'))



# starting date to begin plot
start_date <- as.Date('2019-04-01')

ggplot(df_melted, aes(value, reorder(task, desc(task))), group = task_level) +
  geom_line(size = 5, aes(colour = task_level)) +
  scale_color_manual(values = c('#2164F3', '#99CCFF')) +
  labs(x = '', y = '') +
  theme_bw(base_size = 10) +
  theme(plot.title = element_text(hjust = 0.5),
        axis.text.y = element_text(hjust = 0),
        panel.grid.major.x = element_line(colour="grey"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(angle = 0),
        axis.ticks = element_blank(),
        legend.position = "none") +
  scale_x_date(date_labels = "%b", limits = c(start_date, NA), date_breaks = '1 month') +

  geom_point(aes(x=as.Date('2019-04-30'), y='Project Plan'), colour="red3", size = 3) +
  geom_text(family = 'Helvetica', aes(x=as.Date('2019-04-30'), y='Project Plan',label = 'Proposal due'),hjust=0.5, vjust=-1.2, size = 2.5) +

  geom_point(aes(x=as.Date('2019-12-01'), y='Writeup'), colour="red3", size = 3) +
  geom_text(family = 'Helvetica', aes(x=as.Date('2019-12-01'), y='Writeup',label = 'Project due'),hjust=0.5, vjust=-1.5, size = 2.5) +

  geom_point(aes(x=as.Date('2019-09-01'), y='Writeup'), colour="red3", size = 3) +
  geom_text(family = 'Helvetica', aes(x=as.Date('2019-09-01'), y='Writeup',label = 'First draft'),hjust=0.5, vjust=-1.5, size = 2.5) +

  geom_point(aes(x=as.Date('2019-10-15'), y='Writeup'), colour="red3", size = 3) +
  geom_text(family = 'Helvetica', aes(x=as.Date('2019-10-15'), y='Writeup',label = 'Second draft'),hjust=0.5, vjust=-1.5, size = 2.5) +

  geom_point(aes(x=as.Date('2019-06-01'), y='Experiment 1 :\nSimple Evaluation'), colour="black", size = 3.5, shape = 18) +
  geom_text(family = 'Helvetica', aes(x=as.Date('2019-06-01'), y='Experiment 1 :\nSimple Evaluation',label = 'Experiment kickoff'),hjust=0.5, vjust=-1.5, size = 2.5) +

  geom_point(aes(x=as.Date('2019-07-15'), y='Experiment 2 :\nSystematic error'), colour="black", size = 3.5,shape = 18) +
  geom_text(family = 'Helvetica', aes(x=as.Date('2019-07-15'), y='Experiment 2 :\nSystematic error',label = 'Experiment kickoff'),hjust=0.5, vjust=-1.5, size = 2.5) +

  geom_point(aes(x=as.Date('2019-05-01'), y='Familiarise with\nPython Bayesian ecosystem'), colour="black", size = 3.5, shape = 18) +
  geom_text(family = 'Helvetica', aes(x=as.Date('2019-05-01'), y='Familiarise with\nPython Bayesian ecosystem', label = 'Guidance on data generation'),hjust=0.5, vjust=-1.5, size = 2.5) +

  geom_point(aes(x=as.Date('2019-08-15'), y='Experiment 3 :\nReal dataset'), colour="black", size = 3.5,shape = 18) +
  geom_text(family = 'Helvetica', aes(x=as.Date('2019-08-15'), y='Experiment 3 :\nReal dataset', label = 'Experiment kickoff'),hjust=0.5, vjust=-1.5, size = 2.5) +


  geom_point(aes(x=as.Date('2019-10-01'), y='Experiment 4 :\nReal dataset'), colour="black", size = 3.5,shape = 18) +
  geom_text(family = 'Helvetica', aes(x=as.Date('2019-10-01'), y='Experiment 4 :\nReal dataset', label = 'Experiment kickoff'),hjust=0.5, vjust=-1.5, size = 2.5) +

  geom_point(aes(x=as.Date('2019-11-01'), y='Experiment 4 :\nReal dataset'), colour="black", size = 3.5,shape = 18) +
  geom_text(family = 'Helvetica', aes(x=as.Date('2019-11-01'), y='Experiment 4 :\nReal dataset', label = 'End meeting'),hjust=0.5, vjust=-1.5, size = 2.5) +

  ggsave("/Users/andour/Google Drive/projects/Dissertation/Gannt_chart.png", width = 25, height = 15.15, units = 'cm')





