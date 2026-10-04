# Small-business loan desk: a transparent default scorecard for SBA 7(a) loans.
# Data: U.S. Small Business Administration 7(a) loan-level FOIA data (U.S. Government Works).
library(shiny)
library(bslib)
library(DT)
library(plotly)
library(jsonlite)

card <- fromJSON("data/scorecard.json", simplifyDataFrame = TRUE)
pts <- card$points
pooled <- read.csv("data/q1_pooled.csv")
cohort <- read.csv("data/q1_by_cohort.csv")
deciles <- read.csv("data/q1_deciles.csv")
policy <- read.csv("data/q2_policy_curves.csv")
segments <- read.csv("data/q3_segments.csv")
summ <- fromJSON("data/summary.json")
spread <- fromJSON("data/exploratory_spread.json")

ACCENT <- "#1f6f5c"
GREY <- "#6b7480"
INK <- "#16212b"

sector_names <- c(
  "11" = "Agriculture", "21" = "Mining, oil and gas", "22" = "Utilities", "23" = "Construction",
  "31" = "Manufacturing", "32" = "Manufacturing", "33" = "Manufacturing", "42" = "Wholesale trade",
  "44" = "Retail trade", "45" = "Retail trade", "48" = "Transportation", "49" = "Warehousing",
  "51" = "Information", "52" = "Finance and insurance", "53" = "Real estate", "54" = "Professional services",
  "55" = "Management of companies", "56" = "Administrative and support", "61" = "Educational services",
  "62" = "Health care", "71" = "Arts and recreation", "72" = "Accommodation and food", "81" = "Other services",
  "92" = "Public administration"
)
label_sector <- function(code) ifelse(code %in% names(sector_names), paste0(code, " ", sector_names[code]), code)
pretty <- c(log_amount = "Loan amount", guarantee_share = "SBA guarantee share", interest_rate = "Interest rate",
            jobs = "Jobs supported", log_jobs = "Jobs supported (log)", sector = "Industry", state = "State",
            business_type = "Business type", business_age = "Business age", processing = "Processing method",
            fixed_rate = "Fixed rate", revolver = "Revolving line", collateral = "Collateral",
            franchise = "Franchise")

# Score one application exactly as the Python scorecard does (bins from training data, points per bin).
score_loan <- function(x) {
  feats <- c(card$numeric, card$categorical)
  rows <- lapply(feats, function(f) {
    if (f %in% card$numeric) {
      bin <- as.character(findInterval(x[[f]], card$edges[[f]]))
    } else {
      v <- as.character(x[[f]])
      bin <- if (v %in% card$levels[[f]]) v else "other"
    }
    p <- pts$points[pts$feature == f & pts$bin == bin]
    p <- if (length(p)) p else card$neutral_points
    best <- max(pts$points[pts$feature == f])
    data.frame(feature = f, bin = bin, points = p, shortfall = best - p)
  })
  tab <- do.call(rbind, rows)
  total <- sum(tab$points)
  # Jobs enters the model twice (raw and log); show it as one factor.
  j <- tab$feature %in% c("jobs", "log_jobs")
  tab <- rbind(tab[!j, ], data.frame(feature = "jobs", bin = "", points = sum(tab$points[j]),
                                      shortfall = sum(tab$shortfall[j])))
  pd <- 1 / (1 + exp((total - card$offset) / card$factor))
  list(table = tab, score = total, pd = pd)
}

pct <- function(x, d = 1) sprintf(paste0("%.", d, "f%%"), 100 * x)

about_md <- "
**What this is.** A credit scorecard for SBA 7(a) small-business loans. It shows the default probability
and the reasons behind it, the trade-off of an approval policy, and the checks a model-risk team would run.

**Outcome.** The model predicts whether SBA charged the loan off within five years of approval (early
charge-off). Charge-offs lag defaults, so this captures about 55% of a cohort's eventual charge-offs.

**How it was validated.** Each test cohort (FY2015-FY2020) was scored by a model trained only on cohorts at
least five years older, the outcomes a lender would actually have known. The scorecard on the Loan desk tab is
the same model refit on FY2010-FY2020. Every performance number in this app comes from the held-out cohorts.

**What is not used.** Loan term: SBA overwrites the term of loans that went bad, so it would leak the outcome.
The current servicing bank (loans are transferred after origination). Borrower names and addresses are never
read.

**Data.** U.S. Small Business Administration 7(a) loan-level FOIA data, snapshot as of June 30, 2026
(U.S. Government Works). Prime rate: Federal Reserve Bank of St. Louis (FRED, DPRIME). Not affiliated with
or endorsed by the SBA.

Code and full methods:
[github.com/Andresperez397/sba-loan-default-risk](https://github.com/Andresperez397/sba-loan-default-risk)
"

theme <- bs_theme(version = 5, bg = "#ffffff", fg = INK, primary = ACCENT,
                  base_font = font_collection("Inter", "Helvetica Neue", "Arial", "sans-serif")) |>
  bs_add_rules(".vb-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                          gap: 1rem; margin-bottom: 1rem; }
                .vb-row .bslib-value-box { height: auto !important; min-height: 0 !important; margin: 0; }
                .bslib-value-box .value-box-value { font-size: 1.6rem; }
                .bslib-value-box .value-box-area { padding: 0.6rem 1rem; }")

choices <- card$choices
sector_choices <- setNames(choices$sector, label_sector(choices$sector))

ui <- page_navbar(
  title = "Small-Business Loan Desk",
  theme = theme,
  fillable = FALSE,
  nav_panel(
    "Loan desk",
    layout_sidebar(
      sidebar = sidebar(
        width = 300,
        open = list(desktop = "open", mobile = "always-above"),
        numericInput("amount", "Loan amount ($)", 150000, min = 1000, max = 5e6, step = 5000),
        sliderInput("guarantee", "SBA guarantee share", min = 0.5, max = 0.9, value = 0.75, step = 0.05),
        numericInput("rate", "Initial interest rate (%)", 7.5, min = 2, max = 14, step = 0.25),
        numericInput("jobs", "Jobs supported", 5, min = 0, max = 500),
        selectInput("sector", "Industry (NAICS sector)", sector_choices, selected = "72"),
        selectInput("state", "Project state", choices$state, selected = "TX"),
        selectInput("business_type", "Business type", choices$business_type, selected = "corporation"),
        selectInput("business_age", "Business age", choices$business_age, selected = "existing"),
        selectInput("processing", "Processing method", choices$processing, selected = "preferred_lender"),
        checkboxInput("fixed", "Fixed interest rate", FALSE),
        checkboxInput("revolver", "Revolving line of credit", FALSE),
        checkboxInput("collateral", "Collateral pledged", TRUE),
        checkboxInput("franchise", "Franchise", FALSE)
      ),
      div(class = "vb-row",
          value_box("Score", textOutput("vb_score"), p(class = "small mb-0", "higher = safer"), theme = "light"),
          value_box("Chance of early charge-off", textOutput("vb_pd"), theme = "light"),
          value_box("Compared with all loans", textOutput("vb_pctile"), theme = "light")),
      card(card_header("Top reasons the score is not higher"), uiOutput("reasons")),
      card(card_header("Points by factor"), plotlyOutput("points_plot", height = "420px"))
    )
  ),
  nav_panel(
    "Approval policy",
    layout_columns(
      col_widths = c(4, 8),
      card(
        card_header("Decline the riskiest loans"),
        sliderInput("approve", "Share of applications approved", min = 0.5, max = 1, value = 0.9, step = 0.05),
        uiOutput("policy_text")
      ),
      card(full_screen = TRUE, card_header("Early charge-off rate among approved loans (held-out cohorts FY2015-FY2020)"),
           plotlyOutput("policy_plot", height = "380px"))
    )
  ),
  nav_panel(
    "Model monitoring",
    layout_columns(
      col_widths = c(6, 6),
      card(full_screen = TRUE, card_header("Discrimination by cohort (AUC)"), plotlyOutput("auc_plot", height = "320px")),
      card(full_screen = TRUE, card_header("Calibration by cohort: observed vs predicted"),
           plotlyOutput("cal_plot", height = "320px"))
    ),
    card(card_header("Population stability index (PSI) of each input vs FY2010-FY2015 loans: above 0.25 calls for action"),
         DTOutput("psi_table")),
    card(card_header("Performance by segment (held-out cohorts)"), DTOutput("seg_table"))
  ),
  nav_panel("About", card(markdown(about_md))),
  nav_spacer(),
  nav_item(tags$a("Code", href = "https://github.com/Andresperez397/sba-loan-default-risk", target = "_blank"))
)

server <- function(input, output, session) {
  scored <- reactive({
    x <- list(log_amount = log(max(input$amount, 1)), guarantee_share = input$guarantee,
              interest_rate = input$rate, jobs = input$jobs, log_jobs = log1p(max(input$jobs, 0)),
              sector = input$sector, state = input$state, business_type = input$business_type,
              business_age = input$business_age, processing = input$processing,
              fixed_rate = as.character(as.integer(input$fixed)), revolver = as.character(as.integer(input$revolver)),
              collateral = as.character(as.integer(input$collateral)),
              franchise = as.character(as.integer(input$franchise)))
    score_loan(x)
  })
  output$vb_score <- renderText(sprintf("%.0f", scored()$score))
  output$vb_pd <- renderText(pct(scored()$pd))
  output$vb_pctile <- renderText({
    p <- scored()$pd
    d <- deciles[order(deciles$predicted), ]
    k <- sum(p > d$predicted)
    sprintf("riskier than ~%d%% of loans", 10 * k)
  })
  output$reasons <- renderUI({
    t <- scored()$table
    t <- head(t[order(-t$shortfall), ], 4)
    t <- t[t$shortfall > 0.5, ]
    if (!nrow(t)) return(p("Every factor is already in its best band."))
    tags$ol(lapply(seq_len(nrow(t)), function(i) {
      tags$li(sprintf("%s: %.0f points below its best band", pretty[[t$feature[i]]], t$shortfall[i]))
    }))
  })
  output$points_plot <- renderPlotly({
    t <- scored()$table
    t$label <- factor(pretty[t$feature], levels = rev(pretty[t$feature]))
    plot_ly(t, x = ~points, y = ~label, type = "bar", orientation = "h", marker = list(color = ACCENT),
            text = ~sprintf("%s: %.1f points (%.1f below best)", label, points, shortfall), hoverinfo = "text") |>
      layout(xaxis = list(title = "Points"), yaxis = list(title = "")) |>
      config(displayModeBar = FALSE)
  })

  output$policy_plot <- renderPlotly({
    pl <- policy
    pl$policy <- ifelse(pl$policy == "model", "Decline by scorecard", "Decline by interest rate")
    plot_ly(pl, x = ~approve_share, y = ~default_rate, color = ~policy, colors = c(GREY, ACCENT),
            type = "scatter", mode = "lines+markers",
            text = ~sprintf("%s<br>Approve %s: early charge-off %s, lender loss $%.2f per $100",
                            policy, pct(approve_share, 0), pct(default_rate, 2), lender_loss_per_100),
            hoverinfo = "text") |>
      add_segments(x = input$approve, xend = input$approve, y = 0, yend = max(pl$default_rate),
                   line = list(color = "#c9ced4", dash = "dash"), showlegend = FALSE, inherit = FALSE) |>
      layout(xaxis = list(title = "Share approved", tickformat = ".0%"),
             yaxis = list(title = "Early charge-off rate", tickformat = ".1%"), legend = list(orientation = "h", y = -0.3)) |>
      config(displayModeBar = FALSE)
  })
  output$policy_text <- renderUI({
    q <- round(input$approve, 2)
    m <- policy[policy$policy == "model" & abs(policy$approve_share - q) < 1e-9, ]
    r <- policy[policy$policy == "interest rate" & abs(policy$approve_share - q) < 1e-9, ]
    a <- policy[policy$policy == "model" & policy$approve_share == 1, ]
    tagList(
      p(sprintf("Approving everyone: %s of loans charged off early, lender loss $%.2f per $100 approved.",
                pct(a$default_rate, 2), a$lender_loss_per_100)),
      p(HTML(sprintf("Approving the safest <b>%s</b> by scorecard: %s charged off early, $%.2f per $100.",
                     pct(q, 0), pct(m$default_rate, 2), m$lender_loss_per_100))),
      p(sprintf("By interest rate instead: %s, $%.2f per $100.", pct(r$default_rate, 2), r$lender_loss_per_100)),
      p(class = "text-muted small", "Lender loss = early charge-off amount times the unguaranteed share.
        Declining by rate already removes much of the riskiest tail; the scorecard's edge is in the middle of
        the ranking.")
    )
  })

  output$auc_plot <- renderPlotly({
    x <- cohort[cohort$model %in% c("M1 interest rate", "M2 scorecard", "M3 boosting"), ]
    plot_ly(x, x = ~fy, y = ~auc, color = ~model, colors = c(GREY, ACCENT, "#c27c0e"),
            type = "scatter", mode = "lines+markers") |>
      layout(xaxis = list(title = "Approval fiscal year (test cohort)", dtick = 1), yaxis = list(title = "AUC"),
             legend = list(orientation = "h", y = -0.3)) |>
      config(displayModeBar = FALSE)
  })
  output$cal_plot <- renderPlotly({
    x <- cohort[cohort$model == "M2 scorecard", ]
    plot_ly(x, x = ~fy) |>
      add_trace(y = ~default_rate, name = "Observed", type = "scatter", mode = "lines+markers",
                line = list(color = INK), marker = list(color = INK)) |>
      add_trace(y = ~mean_predicted, name = "Predicted", type = "scatter", mode = "lines+markers",
                line = list(color = ACCENT, dash = "dash"), marker = list(color = ACCENT)) |>
      layout(xaxis = list(title = "Approval fiscal year (test cohort)", dtick = 1),
             yaxis = list(title = "Early charge-off rate", tickformat = ".1%"), legend = list(orientation = "h", y = -0.3)) |>
      config(displayModeBar = FALSE)
  })
  output$psi_table <- renderDT({
    p <- summ$q3$psi_inputs
    tab <- do.call(rbind, lapply(names(p), function(fy) {
      v <- unlist(p[[fy]])
      data.frame(feature = pretty[names(v)], fy = fy, psi = round(v, 3))
    }))
    wide <- reshape(tab, idvar = "feature", timevar = "fy", direction = "wide")
    names(wide) <- sub("psi.", "FY", names(wide), fixed = TRUE)
    datatable(wide, rownames = FALSE, options = list(dom = "t", pageLength = 20)) |>
      formatStyle(names(wide)[-1], backgroundColor = styleInterval(c(0.1, 0.25), c("white", "#fdf1d6", "#f8d5cf")))
  })
  output$seg_table <- renderDT({
    s <- segments[segments$loans >= 1000, ]
    s$level <- ifelse(s$segment == "sector", label_sector(s$level), s$level)
    out <- data.frame(Segment = s$segment, Level = s$level, Loans = s$loans, Defaults = s$defaults,
                      AUC = round(s$auc, 3), Observed = pct(s$observed, 2), Predicted = pct(s$predicted, 2),
                      `Observed / predicted` = round(s$obs_over_pred, 2), check.names = FALSE)
    datatable(out, rownames = FALSE, filter = "top", options = list(pageLength = 12))
  })
}

shinyApp(ui, server)
