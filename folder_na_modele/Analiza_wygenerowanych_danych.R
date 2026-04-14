# ============================================================
# WCZYTANIE BIBLIOTEK I DANYCH
# ============================================================

library(rpart)
library(rpart.plot)
library(ggplot2)
library(dplyr)
library(gridExtra)
library(patchwork)

# Wczytaj 100k wierszy
df <- read.csv("D:\\Projekty\\praca_licencjacka\\Projekt-Formacje-roslinne-na-terenach-pustynniejacych\\folder_na_modele\\wygenerowane_dane_100k.csv")

# Sprawdź dane
cat("=== PODGLĄD DANYCH ===\n")
head(df)
cat("\n=== STRUKTURA DANYCH ===\n")
str(df)

# ============================================================
# DRZEWO DECYZYJNE - ZAPISANE DO PDF (BEZ UPROSZCZANIA)
# ============================================================

# Trenuj drzewo (bez zmian)
tree <- rpart(pattern_name ~ a + m + d1 + d2, 
              data = df, 
              method = "class",
              control = rpart.control(minsplit = 100, cp = 0.001, maxdepth = 10))

# Zapisz do PDF - duży format, wektorowy, można powiększać w nieskończoność
pdf("drzewo_decyzyjne.pdf", width = 40, height = 30)

rpart.plot(tree, 
           type = 2, 
           extra = 104, 
           under = TRUE, 
           fallen.leaves = TRUE,
           cex = 0.6,            # czytelny tekst
           tweak = 0.8,
           gap = 0,
           space = 0,
           main = "Drzewo decyzyjne - przewidywanie wzorów roślinnych")

dev.off()

cat("\n Drzewo zapisane do 'drzewo_decyzyjne.pdf'\n")
cat(" Ścieżka:", getwd(), "\n")

# ============================================================
# NAJWAŻNIEJSZE PARAMETRY (WG DRZEWA)
# ============================================================

cat("\n=== NAJWAŻNIEJSZE PARAMETRY ===\n")
print(tree$variable.importance)

# Wykres ważności parametrów
var_imp <- data.frame(parameter = names(tree$variable.importance),
                      importance = tree$variable.importance)
var_imp <- var_imp[order(-var_imp$importance),]

ggplot(var_imp, aes(x = reorder(parameter, importance), y = importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = round(importance, 2)), hjust = -0.2, size = 4) +
  coord_flip() +
  labs(title = "Ważność parametrów w drzewie decyzyjnym",
       subtitle = "Które parametry mają największy wpływ na przewidywany wzór?",
       x = "Parametr", 
       y = "Ważność") +
  theme_minimal()

# ============================================================
# WARUNKI DLA KAŻDEGO WZORU (PRZEDZIAŁY PARAMETRÓW)
# ============================================================

cat("\n======================================================================\n")
cat("WARUNKI DLA KAŻDEGO WZORU\n")
cat("======================================================================\n")

for(wzor in unique(df$pattern_name)) {
  cat("\n--- ", toupper(wzor), " ---\n", sep = "")
  
  podzbor <- df[df$pattern_name == wzor, ]
  
  cat("  Parametr a (opady):\n")
  cat(paste("    Zakres:", round(min(podzbor$a), 3), "-", round(max(podzbor$a), 3), "\n"))
  cat(paste("    Średnia:", round(mean(podzbor$a), 3), "\n"))
  cat(paste("    Mediana:", round(median(podzbor$a), 3), "\n"))
  
  cat("  Parametr m (śmiertelność):\n")
  cat(paste("    Zakres:", round(min(podzbor$m), 3), "-", round(max(podzbor$m), 3), "\n"))
  cat(paste("    Średnia:", round(mean(podzbor$m), 3), "\n"))
  cat(paste("    Mediana:", round(median(podzbor$m), 3), "\n"))
  
  cat("  Parametr d1 (dyfuzja wody):\n")
  cat(paste("    Zakres:", round(min(podzbor$d1), 3), "-", round(max(podzbor$d1), 3), "\n"))
  cat(paste("    Średnia:", round(mean(podzbor$d1), 3), "\n"))
  cat(paste("    Mediana:", round(median(podzbor$d1), 3), "\n"))
  
  cat("  Parametr d2 (dyfuzja biomasy):\n")
  cat(paste("    Zakres:", round(min(podzbor$d2), 3), "-", round(max(podzbor$d2), 3), "\n"))
  cat(paste("    Średnia:", round(mean(podzbor$d2), 3), "\n"))
  cat(paste("    Mediana:", round(median(podzbor$d2), 3), "\n"))
}

# ============================================================
# REGUŁY DECYZYJNE Z DRZEWA
# ============================================================

cat("\n======================================================================\n")
cat("REGUŁY DECYZYJNE Z DRZEWA\n")
cat("======================================================================\n")

rules <- rpart.rules(tree, cover = TRUE)
print(rules)

# ============================================================
# ANALIZA DLA DZIUR - WSZYSTKIE 4 WYKRESY W JEDNYM OKNIE
# ============================================================

cat("\n======================================================================\n")
cat("JAK UZYSKAĆ DZIURY? - ANALIZA\n")
cat("======================================================================\n")

dziury <- df[df$pattern_name == "dziury", ]

# Statystyki
cat("\n=== STATYSTYKI OPISOWE DLA DZIUR ===\n")
print(summary(dziury[, c("a", "m", "d1", "d2")]))

# 4 WYKRESY W JEDNYM OKNIE - BEZ GRID.ARRANGE (używamy patchwork)
p1 <- ggplot(dziury, aes(x = a)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white", alpha = 0.7) +
  labs(title = "Parametr a (opady)", x = "a", y = "Częstość") +
  theme_minimal()

p2 <- ggplot(dziury, aes(x = m)) +
  geom_histogram(bins = 30, fill = "darkgreen", color = "white", alpha = 0.7) +
  labs(title = "Parametr m (śmiertelność)", x = "m", y = "Częstość") +
  theme_minimal()

p3 <- ggplot(dziury, aes(x = d1)) +
  geom_histogram(bins = 30, fill = "purple", color = "white", alpha = 0.7) +
  labs(title = "Parametr d1 (dyfuzja wody)", x = "d1", y = "Częstość") +
  theme_minimal()

p4 <- ggplot(dziury, aes(x = d2)) +
  geom_histogram(bins = 30, fill = "orange", color = "white", alpha = 0.7) +
  labs(title = "Parametr d2 (dyfuzja biomasy)", x = "d2", y = "Częstość") +
  theme_minimal()

# Wyświetl wszystkie 4 wykresy w jednym oknie (2x2) za pomocą patchwork
(p1 + p2) / (p3 + p4) +
  plot_annotation(title = "Rozkład parametrów dla wzoru DZIURY")

cat("\n======================================================================\n")
cat("JAK UZYSKAĆ PUSTYNIA_LAS? - ANALIZA\n")
cat("======================================================================\n")

pustynia_las <- df[df$pattern_name == "pustynia_las", ]

# Statystyki
cat("\n=== STATYSTYKI OPISOWE DLA PUSTYNIA_LAS ===\n")
print(summary(pustynia_las[, c("a", "m", "d1", "d2")]))

# 5 WYKRESY W JEDNYM OKNIE - BEZ GRID.ARRANGE (używamy patchwork)
p5 <- ggplot(pustynia_las, aes(x = a)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white", alpha = 0.7) +
  labs(title = "Parametr a (opady)", x = "a", y = "Częstość") +
  theme_minimal()

p6 <- ggplot(pustynia_las, aes(x = m)) +
  geom_histogram(bins = 30, fill = "darkgreen", color = "white", alpha = 0.7) +
  labs(title = "Parametr m (śmiertelność)", x = "m", y = "Częstość") +
  theme_minimal()

p7 <- ggplot(pustynia_las, aes(x = d1)) +
  geom_histogram(bins = 30, fill = "purple", color = "white", alpha = 0.7) +
  labs(title = "Parametr d1 (dyfuzja wody)", x = "d1", y = "Częstość") +
  theme_minimal()

p8 <- ggplot(pustynia_las, aes(x = d2)) +
  geom_histogram(bins = 30, fill = "orange", color = "white", alpha = 0.7) +
  labs(title = "Parametr d2 (dyfuzja biomasy)", x = "d2", y = "Częstość") +
  theme_minimal()

# Wyświetl wszystkie 4 wykresy w jednym oknie (2x2) za pomocą patchwork
(p5 + p6) / (p7 + p8) +
  plot_annotation(title = "Rozkład parametrów dla wzoru PUSTYNIA_LAS")

cat("\n======================================================================\n")
cat("JAK UZYSKAĆ PLAMY? - ANALIZA\n")
cat("======================================================================\n")

plamy <- df[df$pattern_name == "plamy", ]

# Statystyki
cat("\n=== STATYSTYKI OPISOWE DLA PLAMY ===\n")
print(summary(plamy[, c("a", "m", "d1", "d2")]))

# 5 WYKRESY W JEDNYM OKNIE - BEZ GRID.ARRANGE (używamy patchwork)
p9 <- ggplot(plamy, aes(x = a)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white", alpha = 0.7) +
  labs(title = "Parametr a (opady)", x = "a", y = "Częstość") +
  theme_minimal()

p10 <- ggplot(plamy, aes(x = m)) +
  geom_histogram(bins = 30, fill = "darkgreen", color = "white", alpha = 0.7) +
  labs(title = "Parametr m (śmiertelność)", x = "m", y = "Częstość") +
  theme_minimal()

p11 <- ggplot(plamy, aes(x = d1)) +
  geom_histogram(bins = 30, fill = "purple", color = "white", alpha = 0.7) +
  labs(title = "Parametr d1 (dyfuzja wody)", x = "d1", y = "Częstość") +
  theme_minimal()

p12 <- ggplot(plamy, aes(x = d2)) +
  geom_histogram(bins = 30, fill = "orange", color = "white", alpha = 0.7) +
  labs(title = "Parametr d2 (dyfuzja biomasy)", x = "d2", y = "Częstość") +
  theme_minimal()

# Wyświetl wszystkie 4 wykresy w jednym oknie (2x2) za pomocą patchwork
(p9 + p10) / (p11 + p12) +
  plot_annotation(title = "Rozkład parametrów dla wzoru PLAMY")

cat("\n======================================================================\n")
cat("JAK UZYSKAĆ LABIRYNTY? - ANALIZA\n")
cat("======================================================================\n")

labirynty <- df[df$pattern_name == "labirynty", ]

# Statystyki
cat("\n=== STATYSTYKI OPISOWE DLA LABIRYNTY ===\n")
print(summary(labirynty[, c("a", "m", "d1", "d2")]))

# 5 WYKRESY W JEDNYM OKNIE - BEZ GRID.ARRANGE (używamy patchwork)
p13 <- ggplot(labirynty, aes(x = a)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white", alpha = 0.7) +
  labs(title = "Parametr a (opady)", x = "a", y = "Częstość") +
  theme_minimal()

p14 <- ggplot(labirynty, aes(x = m)) +
  geom_histogram(bins = 30, fill = "darkgreen", color = "white", alpha = 0.7) +
  labs(title = "Parametr m (śmiertelność)", x = "m", y = "Częstość") +
  theme_minimal()

p15 <- ggplot(labirynty, aes(x = d1)) +
  geom_histogram(bins = 30, fill = "purple", color = "white", alpha = 0.7) +
  labs(title = "Parametr d1 (dyfuzja wody)", x = "d1", y = "Częstość") +
  theme_minimal()

p16 <- ggplot(labirynty, aes(x = d2)) +
  geom_histogram(bins = 30, fill = "orange", color = "white", alpha = 0.7) +
  labs(title = "Parametr d2 (dyfuzja biomasy)", x = "d2", y = "Częstość") +
  theme_minimal()

# Wyświetl wszystkie 4 wykresy w jednym oknie (2x2) za pomocą patchwork
(p13 + p14) / (p15 + p16) +
  plot_annotation(title = "Rozkład parametrów dla wzoru LABIRYNTY")

cat("\n======================================================================\n")
cat("JAK UZYSKAĆ INNE? - ANALIZA\n")
cat("======================================================================\n")

inne <- df[df$pattern_name == "inne", ]

# Statystyki
cat("\n=== STATYSTYKI OPISOWE DLA INNE ===\n")
print(summary(inne[, c("a", "m", "d1", "d2")]))

# 5 WYKRESY W JEDNYM OKNIE - BEZ GRID.ARRANGE (używamy patchwork)
p17 <- ggplot(inne, aes(x = a)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "white", alpha = 0.7) +
  labs(title = "Parametr a (opady)", x = "a", y = "Częstość") +
  theme_minimal()

p18 <- ggplot(inne, aes(x = m)) +
  geom_histogram(bins = 30, fill = "darkgreen", color = "white", alpha = 0.7) +
  labs(title = "Parametr m (śmiertelność)", x = "m", y = "Częstość") +
  theme_minimal()

p19 <- ggplot(inne, aes(x = d1)) +
  geom_histogram(bins = 30, fill = "purple", color = "white", alpha = 0.7) +
  labs(title = "Parametr d1 (dyfuzja wody)", x = "d1", y = "Częstość") +
  theme_minimal()

p20 <- ggplot(inne, aes(x = d2)) +
  geom_histogram(bins = 30, fill = "orange", color = "white", alpha = 0.7) +
  labs(title = "Parametr d2 (dyfuzja biomasy)", x = "d2", y = "Częstość") +
  theme_minimal()

# Wyświetl wszystkie 4 wykresy w jednym oknie (2x2) za pomocą patchwork
(p17 + p18) / (p19 + p20) +
  plot_annotation(title = "Rozkład parametrów dla wzoru INNE")

# ============================================================
# REGRESJA LOGISTYCZNA DLA KAŻDEGO WZORU
# ============================================================

cat("\n======================================================================\n")
cat("REGRESJA LOGISTYCZNA - WPŁYW PARAMETRÓW\n")
cat("======================================================================\n")

for(wzor in unique(df$pattern_name)) {
  cat("\n--- ", wzor, " vs reszta ---\n", sep = "")
  
  df$target <- ifelse(df$pattern_name == wzor, 1, 0)
  
  log_model <- glm(target ~ a + m + d1 + d2, 
                   data = df, 
                   family = binomial)
  
  summary_log <- summary(log_model)
  cat("\nWspółczynniki regresji:\n")
  print(summary_log$coefficients)
  
  coefs <- summary_log$coefficients[-1, 1]
  names(coefs) <- c("a", "m", "d1", "d2")
  
  cat("\nInterpretacja:\n")
  cat(paste("  Zwiększa szansę na", wzor, ":", names(which.max(coefs)), "\n"))
  cat(paste("  Zmniejsza szansę na", wzor, ":", names(which.min(coefs)), "\n"))
}

# ============================================================
# ODPOWIEDŹ DLA GEOLOGA - PRZEDZIAŁY PARAMETRÓW
# ============================================================

cat("\n======================================================================\n")
cat("ODPOWIEDŹ DLA GEOLOGA - PRZEDZIAŁY PARAMETRÓW\n")
cat("======================================================================\n")

for(wzor in unique(df$pattern_name)) {
  czyste <- df[df$pattern_name == wzor, ]
  
  cat("\n========================================\n")
  cat(toupper(wzor), "\n")
  cat("========================================\n")
  cat("Aby uzyskać ten wzór, ustaw:\n")
  cat(paste("  a (opady):", round(quantile(czyste$a, 0.05), 3), "-", round(quantile(czyste$a, 0.95), 3), "\n"))
  cat(paste("  m (śmiertelność):", round(quantile(czyste$m, 0.05), 3), "-", round(quantile(czyste$m, 0.95), 3), "\n"))
  cat(paste("  d1 (dyfuzja wody):", round(quantile(czyste$d1, 0.05), 3), "-", round(quantile(czyste$d1, 0.95), 3), "\n"))
  cat(paste("  d2 (dyfuzja biomasy):", round(quantile(czyste$d2, 0.05), 3), "-", round(quantile(czyste$d2, 0.95), 3), "\n"))
}

# ============================================================
# ZAPISZ REGUŁY DO PLIKU
# ============================================================

write.csv(rules, "reguly_dezyzyjne.csv", row.names = FALSE)
cat("\n Reguły zapisane do 'reguly_dezyzyjne.csv'\n")

# ============================================================
# WYKRES ROZKŁADU WZORÓW (JEDEN WYKRES)
# ============================================================

# Wykres a vs m
ggplot(df %>% sample_n(5000), aes(x = a, y = m, color = pattern_name)) +
  geom_point(alpha = 0.5, size = 1) +
  labs(title = "Rozkład wzorów - opady (a) vs śmiertelność (m)",
       x = "a (opady)", y = "m (śmiertelność)", color = "Wzór") +
  theme_minimal()

cat("\n Analiza zakończona!\n")