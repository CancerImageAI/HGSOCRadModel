setwd(getwd())
library(rms)
library(pROC)
library(survival)
library(survminer)
library(survcomp)
library(rmeta)

FUSCC_data <- read.csv("./Result/Training_PredictionResult.csv")
Stage <- FUSCC_data$stage
Stage[Stage=='III'|Stage=='IIIa'|Stage=='IIIb'|Stage=='IIIc']=0
Stage[Stage=='IVa'|Stage=='IVb'|Stage=='IV']=1
Stage<-as.numeric(Stage)
Age <- FUSCC_data$age
Age[Age<=60] = 0
Age[Age>60] = 1
Rad_Score <- FUSCC_data$prob
Residual <- FUSCC_data$residual
Residual[Residual=="R0"] = 0
Residual[Residual=="R1"] = 1
Residual[Residual=="R2"] = 2
Residual <- as.numeric(Residual)
Mass <- FUSCC_data$mass_characteristic-1
Bilaterality <- FUSCC_data$bilaterality-1
Diameter <- FUSCC_data$Diameter
CA125 <- FUSCC_data$CA125
CA125[CA125<=35]=0
CA125[CA125>35]=1
Resistance <- FUSCC_data$resistance
FUSCC_DF <- data.frame(Rad_Score,Age,Stage,Residual,Mass, Bilaterality,Diameter,CA125,Resistance)
FUSCC_DF$Age <- factor(FUSCC_DF$Age, level = c(0,1), labels = c('<=60','>60'))
FUSCC_DF$Mass <- factor(FUSCC_DF$Mass,levels = c(0,1,2), labels = c('C','M','S'))
FUSCC_DF$Bilaterality <- factor(FUSCC_DF$Bilaterality, levels = c(0,1),labels = c("Yes","No"))
FUSCC_DF$Diameter <- factor(FUSCC_DF$Diameter, levels = c(0,1), labels = c("<5mm",">=5mm"))
FUSCC_DF$Stage <- factor(FUSCC_DF$Stage, levels = c(0,1), labels = c("III","IV"))
FUSCC_DF$Residual <- factor(FUSCC_DF$Residual, levels = c(0,1,2), labels = c("R0","R1","R2"))
FUSCC_DF$CA125 <- factor(FUSCC_DF$CA125, levels = c(0,1), labels = c('-','+'))
FUSCC_dd = datadist(FUSCC_DF)
options(datadist="FUSCC_dd")


clf_model <- lrm(Resistance~Rad_Score+Age+Stage+Residual+Mass+Bilaterality+Diameter+CA125,data=FUSCC_DF,x=TRUE,y=TRUE,tol=1e-9,maxit=2000)
nom <- nomogram(clf_model,  fun=plogis, 
                fun.at=c(0.001,0.5,0.999),
                lp=F, funlabel="Risk of Resistance")
plot(nom, xfrac=0.1,#1.变量与图形的占比     
     cex.var=0.8, #2.变量字体加粗
     cex.axis=0.7,#3.数轴：字体的大小
     tcl=-0.2, #4.数轴：刻度的长度
     lmgp=0.3,#5.数轴：文字与刻度的距离
     label.every=1,#6.数轴：刻度下的文字，1=连续显示，2=隔一个显示一个
     dec = 1,
     #7.1个页面有几个数轴(这个可以压缩行间距)naxes=20,
     # col.grid=c("Tomato2","DodgerBlue"),#8.垂直线的颜色.
     lplabel="Linear Predictorlp",#9.线性预测轴名字
     points.label='Points', #10变量分数名字
     total.points.label='Total Points',   #11总分名字
     force.label=T,#没啥用TRUE强制标记的每个刻度线都绘制标签，我也没研究明白col.grid = c("Tomato2","DodgerBlue")
)
FUSCC_pred <- predict(clf_model,newdata = FUSCC_DF)
plot.roc(FUSCC_DF$Resistance, FUSCC_pred,print.thres = TRUE,
         main="ROC Curve", percent=TRUE,
         print.auc=TRUE,
         ci=TRUE)
FUSCC_roc <- roc(FUSCC_DF$Resistance, FUSCC_pred)
FUSCC_result <- coords(FUSCC_roc, "best")
FUSCC_group <- as.numeric(FUSCC_pred>=FUSCC_result$threshold)
CMP_FUSCC = survfit(Surv(PFS_time, resistance)~FUSCC_group,data = FUSCC_data)
summary(CMP_FUSCC)
survdiff(formula=Surv(PFS_time, resistance)~FUSCC_group,data = FUSCC_data)

ggsurvplot(
  CMP_FUSCC,                     # survfit object with calculated statistics.
  pval = TRUE,             # show p-value of log-rank test.
  conf.int = TRUE,         # show confidence intervals for 
  # point estimaes of survival curves.
  #conf.int.style = "step",  # customize style of confidence intervals
  xlab = "Survival Time (Month)",   # customize X axis label.
  break.time.by = 6,     # break X axis in time intervals by 200.
  #ggtheme = theme_light(), # customize plot and risk table with a theme.
  risk.table = TRUE,  # absolute number and percentage at risk.
  risk.table.y.text.col = T,# colour risk table text annotations.
  risk.table.y.text = FALSE,# show bars instead of names in text annotations
  risk.table.height = .3,
  # in legend of risk table.
  ncensor.plot = FALSE,      # plot the number of censored subjects at time t
  surv.median.line = "hv",  # add the median survival pointer.
  legend.title = "Training Dataset",
  legend.labs = 
    c('Low risk','High risk'),    # change legend labels.
  palette = 
    c("dodgerblue2", "orchid2") # custom color palettes.
)

FUSCC_data$OS_status[FUSCC_data$OS_status==2]=0
CMP_FUSCC = survfit(Surv(OS_time, OS_status)~FUSCC_group,data = FUSCC_data)
summary(CMP_FUSCC)
survdiff(formula=Surv(OS_time, OS_status)~FUSCC_group,data = FUSCC_data)

ggsurvplot(
  CMP_FUSCC,                     # survfit object with calculated statistics.
  pval = TRUE,             # show p-value of log-rank test.
  conf.int = TRUE,         # show confidence intervals for 
  # point estimaes of survival curves.
  #conf.int.style = "step",  # customize style of confidence intervals
  xlab = "Survival Time (Month)",   # customize X axis label.
  break.time.by = 6,     # break X axis in time intervals by 200.
  #ggtheme = theme_light(), # customize plot and risk table with a theme.
  risk.table = TRUE,  # absolute number and percentage at risk.
  risk.table.y.text.col = T,# colour risk table text annotations.
  risk.table.y.text = FALSE,# show bars instead of names in text annotations
  risk.table.height = .3,
  # in legend of risk table.
  ncensor.plot = FALSE,      # plot the number of censored subjects at time t
  surv.median.line = "hv",  # add the median survival pointer.
  legend.title = "FUSCC_OS",
  legend.labs = 
    c('Low risk','High risk'),    # change legend labels.
  palette = 
    c("dodgerblue2", "orchid2") # custom color palettes.
)



Test_data <- read.csv("./Result/Testing_PredictionResult.csv")
Stage <- Test_data$stage
Stage[Stage=='III'|Stage=='IIIa'|Stage=='IIIb'|Stage=='IIIc']=0
Stage[Stage=='IVa'|Stage=='IVb'|Stage=='IV']=1
Stage<-as.numeric(Stage)
Age <- Test_data$age
Age[Age<=60] = 0
Age[Age>60] = 1
Rad_Score <- Test_data$prob
Residual <- Test_data$residual
Residual[Residual=="R0"] = 0
Residual[Residual=="R1"] = 1
Residual[Residual=="R2"] = 2
Residual <- as.numeric(Residual)
Mass <- Test_data$mass_characteristic-1
Bilaterality <- Test_data$bilaterality-1
Diameter <- Test_data$Diameter
CA125 <- Test_data$CA125
CA125[CA125<=35]=0
CA125[CA125>35]=1
Resistance <- Test_data$resistance
Test_DF <- data.frame(Rad_Score,Age,Stage,Residual,Mass, Bilaterality,Diameter,CA125,Resistance)
Test_DF$Age <- factor(Test_DF$Age, level = c(0,1), labels = c('<=60','>60'))
Test_DF$Mass <- factor(Test_DF$Mass,levels = c(0,1,2), labels = c('C','M','S'))#'Cystic','Mixed','Solid'
Test_DF$Bilaterality <- factor(Test_DF$Bilaterality, levels = c(0,1),labels = c("Yes","No"))
Test_DF$Diameter <- factor(Test_DF$Diameter, levels = c(0,1), labels = c("<5mm",">=5mm"))
Test_DF$Stage <- factor(Test_DF$Stage, levels = c(0,1), labels = c("III","IV"))
Test_DF$Residual <- factor(Test_DF$Residual, levels = c(0,1,2), labels = c("R0","R1","R2"))
Test_DF$CA125 <- factor(Test_DF$CA125, levels = c(0,1), labels = c('-','+'))

Test_pred <- predict(clf_model,newdata = Test_DF)

roc1 <- plot.roc(Test_DF$Resistance, Test_pred,
                 main="Testing Dataset", 
                 print.auc=TRUE,legacy.axes = TRUE,
                 ci=TRUE,col = c("#1c61b6"),lwd = 2)

roc_radscore <- plot.roc(Test_DF$Resistance, Rad_Score,
                 main="Testing Dataset-Rad", 
                 print.auc=TRUE,legacy.axes = TRUE,
                 ci=TRUE,col = c("#1c61b6"),lwd = 2)
Clinical_Score <- Test_data$clinical_prob
roc_clinical <- plot.roc(Test_DF$Resistance, Clinical_Score,
                         main="Testing Dataset-Clinical", 
                         print.auc=TRUE,legacy.axes = TRUE,
                         ci=TRUE,col = c("#1c61b6"),lwd = 2)

roc.test(roc1,roc_radscore,method = 'delong')
roc.test(roc1,roc_clinical,method = 'delong')
roc.test(roc_radscore,roc_clinical,method = 'delong')

Test_roc <- roc(Test_DF$Resistance, Test_pred)
Test_result <- coords(Test_roc, "best")
Test_group <- as.numeric(Test_pred>=Test_result$threshold)
CMP_Test = survfit(Surv(PFS_time, resistance)~Test_group,data = Test_data)
summary(CMP_Test)
survdiff(formula=Surv(PFS_time, resistance)~Test_group,data = Test_data)

ggsurvplot(
  CMP_Test,                     # survfit object with calculated statistics.
  pval = TRUE,             # show p-value of log-rank test.
  conf.int = TRUE,         # show confidence intervals for 
  # point estimaes of survival curves.
  #conf.int.style = "step",  # customize style of confidence intervals
  xlab = "Survival Time (Month)",   # customize X axis label.
  break.time.by = 6,     # break X axis in time intervals by 200.
  #ggtheme = theme_light(), # customize plot and risk table with a theme.
  risk.table = TRUE,  # absolute number and percentage at risk.
  risk.table.y.text.col = T,# colour risk table text annotations.
  risk.table.y.text = FALSE,# show bars instead of names in text annotations
  risk.table.height = .3,
  # in legend of risk table.
  ncensor.plot = FALSE,      # plot the number of censored subjects at time t
  surv.median.line = "hv",  # add the median survival pointer.
  legend.title = "Testing Dataset",
  legend.labs = 
    c('Low risk','High risk'),    # change legend labels.
  palette = 
    c("dodgerblue2", "orchid2") # custom color palettes.
)

Test_data$OS_status[Test_data$OS_status==2]=0
CMP_Test = survfit(Surv(OS_time, OS_status)~Test_group,data = Test_data)
summary(CMP_Test)
survdiff(formula=Surv(OS_time, OS_status)~Test_group,data = Test_data)

ggsurvplot(
  CMP_Test,                     # survfit object with calculated statistics.
  pval = TRUE,             # show p-value of log-rank test.
  conf.int = TRUE,         # show confidence intervals for 
  xlab = "Survival Time (Month)",   # customize X axis label.
  break.time.by = 6,     # break X axis in time intervals by 200.
  #ggtheme = theme_light(), # customize plot and risk table with a theme.
  risk.table = TRUE,  # absolute number and percentage at risk.
  risk.table.y.text.col = T,# colour risk table text annotations.
  risk.table.y.text = FALSE,# show bars instead of names in text annotations
  risk.table.height = .3,
  # in legend of risk table.
  ncensor.plot = FALSE,      # plot the number of censored subjects at time t
  surv.median.line = "hv",  # add the median survival pointer.
  legend.title = "Test_OS",
  legend.labs = 
    c('Low risk','High risk'),    # change legend labels.
  palette = 
    c("dodgerblue2", "orchid2") # custom color palettes.
)
