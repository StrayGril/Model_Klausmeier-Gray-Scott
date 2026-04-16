# ============================================================
# FINALNA ANALIZA
# ============================================================

library(rpart)
library(rpart.plot)
library(ggplot2)
library(dplyr)
library(patchwork)

# Wczytaj dane
df <- read.csv("D:\\Projekty\\praca_licencjacka\\Projekt-Formacje-roslinne-na-terenach-pustynniejacych\\folder_na_modele\\wygenerowane_dane_100k.csv")

# ============================================================
# 1. DRZEWO
# ============================================================

tree <- rpart(pattern_name ~ a + m + d1 + d2,
              data = df,
              method = "class",
              control = rpart.control(
                maxdepth = 6,
                minsplit = 200,
                minbucket = 100,
                cp = 0.001
              ))

# Zapisz drzewo

png("drzewo.png", width = 5000, height = 4000, res = 300)
rpart.plot(tree, 
           type = 2, 
           extra = 104, 
           under = TRUE, 
           fallen.leaves = FALSE,
           cex = 0.8, 
           tweak = 1.0, 
           gap = 0.5, 
           space = 0.5, 
           branch = 0.5,
           main = "Drzewo decyzyjne - przewidywanie wzorów roślinnych")
dev.off()
cat("Zapisano 'drzewo.png'\n")

# ============================================================
# 2. WYKRES a vs m
# ============================================================

# Definiuj kolory dla każdego wzoru
kolory <- c(
  "dziury" = "#D62728",        # CZERWONY
  "inne" = "#FF7F0E",          # POMARAŃCZOWY
  "labirynty" = "#2CA02C",     # ZIELONY
  "plamy" = "#1F77B4",         # NIEBIESKI
  "pustynia_las" = "#9467BD"   # FIOLETOWY
)

p1 <- ggplot(df %>% sample_n(10000), aes(x = a, y = m, color = pattern_name)) +
  geom_point(alpha = 0.6, size = 1.5) +
  scale_color_manual(values = kolory, name = "Wzór") +
  labs(title = "Rozkład wzorów - opady (a) vs śmiertelność (m)",
       x = "a (opady)", 
       y = "m (śmiertelność)") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5))

ggsave("wykres_a_vs_m.png", plot = p, width = 10, height = 8, dpi = 300)
cat("Zapisano 'wykres_a_vs_m.png'\n")

p1

# ============================================================
# 3. INSTRUKCJA OBSŁUGI DRZEWA
# ============================================================

cat("\n========================================================\n")
cat("JAK CZYTAĆ DRZEWO:\n")
cat("========================================================\n")
cat("1. Zaczynasz od góry (korzeń)\n")
cat("2. Zadajesz pytanie - jeśli TAK idziesz w lewo, jeśli NIE w prawo\n")
cat("3. Powtarzasz aż dojdziesz do liścia (prostokąt)\n")
cat("4. W liściu odczytujesz przewidywany wzór\n")
cat("\nPrzykład:\n")
cat("  a = 1.0 -> lewo -> PUSTYNIA\n")
cat("  a = 2.5, m = 0.4 -> prawo -> prawo -> PLAMY\n")
cat("  a = 3.0, m = 0.2 -> prawo -> lewo -> LABIRYNT\n")

cat("\n Gotowe!\n")