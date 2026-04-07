library(tidyverse)
library(glmmTMB)
library(tidyr)
library(dplyr)

setwd('~/Desktop/HVRD/workspace/manifold-dynamics/aux_controls/')

# load in dataframes
ed_main <- read.csv('sampling_strength_summary.csv')
ed_long <- read.csv('zscore_sampling_strength_summary.csv')

ed_long <- ed_long %>%
  left_join(
    ed_main %>% select(roi_key, avg_topk_zscore),
    by = c("target" = "roi_key")
  )
df_wide <- ed_long %>%
  select(target, condition, ED_z, avg_topk_zscore) %>%
  pivot_wider(
    names_from = condition,
    values_from = ED_z,
    names_prefix = "EDz_"
  )

p03 <- ggplot(df_wide, aes(x = avg_topk_zscore, y = EDz_local)) +
  geom_point() + 
  labs(title="Local ED (z) vs average z", y="Local ED (z)", x="Top-k avg z")
p03


# distribution of local ED values
p00 <- ggplot(ed_long, aes(x=ED_z, color=condition)) + 
  geom_density() + 
  labs(title="Distribution of local ED")
p00
# ggsave("local_distribution.png", plot = p00, width = 3, height = 2, dpi = 300)

# sampling strength vs local ED
p02 <- ggplot(ed_main, aes(x=avg_topk_zscore, y=local_ed)) + 
  geom_point() + 
  labs(title="Sampling strength vs Local ED", x="Average top-k z-score", y="Local ED")
ggsave("scatter_zscore_local.png", plot = p02, width = 3, height = 2, dpi = 300)

str(ed_long)
# turn variables into factors as needed
ed_long$condition <- as.factor(ed_long$condition)
ed_long$target_group <- as.factor(ed_long$target_group)
ed_long$target <- as.factor(ed_long$target)
str(ed_long)

# histogram of ED values
p1 <- ggplot(ed_long, aes(x=ED)) + 
  geom_density()
ggsave("pooled_distribution.png", plot = p1, width = 3, height = 2, dpi = 300)

# slightly right skewed, strictly positive
p2 <- ggplot(ed_long, aes(x=ED, color=type)) + 
  geom_density() + 
  labs(title="ED marginal distributions")
ggsave("marginal_distribution.png", plot = p2, width = 4, height = 2, dpi = 300)

# by major selectivity
p3 <- ggplot(ed_long, aes(x=major_selectivity, y=ED, color=type)) + 
  geom_boxplot() + 
  labs(title="ED by selectivity")
ggsave("ed_boxplot01.png", plot = p3, width = 4, height = 3, dpi = 300)

p4 <- ggplot(ed_long, aes(x=type, y=ED, color=major_selectivity)) + 
  geom_boxplot() + 
  labs(title="ED by selectivity")
ggsave("ed_boxplot02.png", plot = p4, width = 4, height = 3, dpi = 300)

# simple glmm fit
fit03.1 <- glmmTMB(data=ed_long, ED ~ type)
summary(fit03.1)
# no difference in ED across 3 ROIs
fit03.2 <- glmmTMB(data=ed_long, ED ~ type + major_selectivity)
summary(fit03.2)
# not what this means. face ROIs have larger ED in general?
fit03.3 <- glmmTMB(data=ed_long, ED ~ type * major_selectivity)
summary(fit03.3)

# interaction between ED change and top-k sampling strength
# both effects are present: local --> global and sampling strength --> low ED
fit04 <- glmmTMB(data=ed_long, ED ~ type * avg_topk_zscore)
summary(fit04)

# keeping it maximal
fit05 <- glmmTMB(data=ed_long, ED ~ type * avg_topk_zscore * major_selectivity)
summary(fit05)
