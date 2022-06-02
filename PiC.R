#Download the NPRED package from "https://www.hydrology.unsw.edu.au/download/software/npred" and then install the library
#install.packages("C:/Users/shubh/Desktop/PIC/NPRED_1.0.1.zip", repos = NULL, type = "win.binary")
setwd("C:/Users/shubh/Desktop/PIC/")
library(NPRED)
library(csvread)
library(readxl)
library(parallel)
library(foreach)
library(doParallel)

####Global Dataset ICRAF
IP <- read.csv("VNIR.csv")
#IP <- read.csv("MIR.csv")
IP=as.data.frame(IP)
mapping <- c("SiCl" = 1, "Cl" = 2, "SiLo" = 3, "Lo" = 4, "SaLo" = 5, "SiClLo" = 6, 
             "LoSa" = 7, "Sa" = 8, "Si" = 9, "SaClLo" = 10, "ClLo" = 11, "SaCl" = 12 )
IP$Texture <- mapping[IP$Texture]
colnames(IP)

stratified <- function(df, group, size, select = NULL, 
                        replace = FALSE, bothSets = FALSE) {
  if (is.null(select)) {
    df <- df
  } else {
    if (is.null(names(select))) stop("'select' must be a named list")
    if (!all(names(select) %in% names(df)))
      stop("Please verify your 'select' argument")
    temp <- sapply(names(select),
                   function(x) df[[x]] %in% select[[x]])
    df <- df[rowSums(temp) == length(select), ]
  }
  df.interaction <- interaction(df[group], drop = TRUE)
  df.table <- table(df.interaction)
  df.split <- split(df, df.interaction)
  if (length(size) > 1) {
    if (length(size) != length(df.split))
      stop("Number of groups is ", length(df.split),
           " but number of sizes supplied is ", length(size))
    if (is.null(names(size))) {
      n <- setNames(size, names(df.split))
      message(sQuote("size"), " vector entered as:\n\nsize = structure(c(",
              paste(n, collapse = ", "), "),\n.Names = c(",
              paste(shQuote(names(n)), collapse = ", "), ")) \n\n")
    } else {
      ifelse(all(names(size) %in% names(df.split)),
             n <- size[names(df.split)],
             stop("Named vector supplied with names ",
                  paste(names(size), collapse = ", "),
                  "\n but the names for the group levels are ",
                  paste(names(df.split), collapse = ", ")))
    }
  } else if (size < 1) {
    n <- round(df.table * size, digits = 0)
  } else if (size >= 1) {
    if (all(df.table >= size) || isTRUE(replace)) {
      n <- setNames(rep(size, length.out = length(df.split)),
                    names(df.split))
    } else {
      message(
        "Some groups\n---",
        paste(names(df.table[df.table < size]), collapse = ", "),
        "---\ncontain fewer observations",
        " than desired number of samples.\n",
        "All observations have been returned from those groups.")
      n <- c(sapply(df.table[df.table >= size], function(x) x = size),
             df.table[df.table < size])
    }
  }
  temp <- lapply(
    names(df.split),
    function(x) df.split[[x]][sample(df.table[x],
                                     n[x], replace = replace), ])
  set1 <- do.call("rbind", temp)
  
  if (isTRUE(bothSets)) {
    set2 <- df[!rownames(df) %in% rownames(set1), ]
    list(SET1 = set1, SET2 = set2)
  } else {
    set1
  }
}
  

no_cores <- detectCores()
# Setup cluster
clust <- makeCluster(no_cores-2) #This line will take time
registerDoParallel(clust)
clusterExport(clust, "IP")


s = system.time({foo1 = foreach(i=1:10, .combine = c, .packages="NPRED")  %dopar%  
  {stratified <- function(df, group, size, select = NULL, 
                          replace = FALSE, bothSets = FALSE) {
    if (is.null(select)) {
      df <- df
    } else {
      if (is.null(names(select))) stop("'select' must be a named list")
      if (!all(names(select) %in% names(df)))
        stop("Please verify your 'select' argument")
      temp <- sapply(names(select),
                     function(x) df[[x]] %in% select[[x]])
      df <- df[rowSums(temp) == length(select), ]
    }
    df.interaction <- interaction(df[group], drop = TRUE)
    df.table <- table(df.interaction)
    df.split <- split(df, df.interaction)
    if (length(size) > 1) {
      if (length(size) != length(df.split))
        stop("Number of groups is ", length(df.split),
             " but number of sizes supplied is ", length(size))
      if (is.null(names(size))) {
        n <- setNames(size, names(df.split))
        message(sQuote("size"), " vector entered as:\n\nsize = structure(c(",
                paste(n, collapse = ", "), "),\n.Names = c(",
                paste(shQuote(names(n)), collapse = ", "), ")) \n\n")
      } else {
        ifelse(all(names(size) %in% names(df.split)),
               n <- size[names(df.split)],
               stop("Named vector supplied with names ",
                    paste(names(size), collapse = ", "),
                    "\n but the names for the group levels are ",
                    paste(names(df.split), collapse = ", ")))
      }
    } else if (size < 1) {
      n <- round(df.table * size, digits = 0)
    } else if (size >= 1) {
      if (all(df.table >= size) || isTRUE(replace)) {
        n <- setNames(rep(size, length.out = length(df.split)),
                      names(df.split))
      } else {
        message(
          "Some groups\n---",
          paste(names(df.table[df.table < size]), collapse = ", "),
          "---\ncontain fewer observations",
          " than desired number of samples.\n",
          "All observations have been returned from those groups.")
        n <- c(sapply(df.table[df.table >= size], function(x) x = size),
               df.table[df.table < size])
      }
    }
    temp <- lapply(
      names(df.split),
      function(x) df.split[[x]][sample(df.table[x],
                                       n[x], replace = replace), ])
    set1 <- do.call("rbind", temp)
    
    if (isTRUE(bothSets)) {
      set2 <- df[!rownames(df) %in% rownames(set1), ]
      list(SET1 = set1, SET2 = set2)
    } else {
      set1
    }
  }
  IP1 = stratified(IP, "Texture", .75)
  x<-IP1["Texture"]   
  Y<-IP1[,2:ncol(IP1)]
  x1 = 1 
  lband = list(10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,204)
  #lband = list(88,176,264,352,440,528,616,704,792,880,968,1056,1144,1232,1320,1408,1496,1584,1672,1762)
  all_imp_bands = NULL
  for (x2 in lband){
    py<-Y[,x1:x2]  # possible predictors
    result = stepwise.PIC(x,py)
    if (x2 == lband[1]) {
      imp_bands = result$cpy
    } else {
      imp_bands = (x1 - 1) + result$cpy
    }
    all_imp_bands = c(all_imp_bands,imp_bands)
    x1 = x2 + 1
  }
  names <- colnames(Y)
  names1 = names[all_imp_bands]
  names1
  }
})


stopCluster(clust)
rm(foo1,s)
stopImplicitCluster()




##Original without parallel implementation
#response
set.seed(1)
all_imp_bands1 = NULL
for (i in 1:10) {
  IP1 = stratified(IP, "Texture", .75)
  x<-IP1["Texture"]   
  Y<-IP1[,2:ncol(IP)]
  #colnames(Y)
  x1 = 1 
  lband = list(3792)
  #lband = list(45,100,131,191,260)
  #lband = list(21,45,87,100,116,131,156,191,226,260)
  #lband = list(15,30,45,63,81,100,110,120,131,151,171,191,214,237,260)
  #lband = list(11,22,33,45,59,73,87,100,108,116,124,131,146,161,176,191,208,225,242,260)
  all_imp_bands = NULL
  for (x2 in lband){
    py<-Y[,x1:x2]  # possible predictors
    result = stepwise.PIC(x,py)
    if (x2 == lband[1]) {
      imp_bands = result$cpy
    } else {
      imp_bands = (x1 - 1) + result$cpy
    }
    all_imp_bands = c(all_imp_bands,imp_bands)
    x1 = x2 + 1
  }
  all_imp_bands1 = c(all_imp_bands1,all_imp_bands)
}
all_imp_bands1
names <- colnames(Y)
names1 = names[all_imp_bands1]
names1