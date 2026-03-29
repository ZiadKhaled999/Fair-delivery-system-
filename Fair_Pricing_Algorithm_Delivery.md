# Developing a Fair, AI-Augmented Dynamic Pricing Algorithm for On-Demand Last-Mile Delivery Services: Evidence-Based Design for Emerging Urban Markets

**Authors:** [Ziad Khaled Mohamed]
**Correspondence:** [albhyrytwamrwhy@gmail.com]
**Date:** March 2026
**Keywords:** Dynamic Pricing, Last-Mile Delivery, Algorithmic Fairness, Machine Learning, Gig Economy, Emerging Markets, Natural Language Processing, Demand Forecasting

---

## Abstract

The proliferation of on-demand food and parcel delivery platforms in rapidly urbanizing economies has exposed a fundamental tension between platform profitability, courier equity, and consumer welfare. Existing pricing architectures in developing markets — including Egypt — predominantly rely on static fee structures or rudimentary distance-based heuristics, which systematically fail to capture the multidimensional cost structure of last-mile delivery operations. This paper presents the design, formalization, and empirical grounding of a Fair Dynamic Pricing Algorithm (FDPA) for on-demand delivery platforms, constructed upon a tripartite taxonomy of objective, measurable determinants: *spatial-geographic*, *temporal-seasonal*, and *operational* variables.

The proposed model integrates an AI-augmented multiplier component that continuously synthesizes macroeconomic indicators (fuel price volatility, consumer price index), sentiment signals derived from Arabic-language NLP pipelines, and a predictive demand model trained on historical order-flow data. The core pricing equation is formalized as:

$$\text{Fare} = B \cdot \alpha_t + (D \times R_d \times \beta_{\text{traffic}}) + (W_r \times R_w \times \gamma_{\text{rest}}) + (A \times R_a) + M_{\text{AI}}$$

A Mixed Methods research design — combining quantitative regression analysis over a target corpus of 2,000+ georeferenced orders with qualitative thematic analysis of courier and consumer interviews — is employed to empirically calibrate model coefficients. The FDPA is intentionally decoupled from courier-specific attributes (vehicle type, age, personal rating history), operationalizing a task-centric rather than agent-centric fairness paradigm. The paper contributes (i) a formally grounded pricing model contextualized for Egyptian urban logistics, (ii) a reproducible methodology for coefficient calibration in data-sparse emerging markets, and (iii) a multi-stakeholder fairness framework applicable to gig economy platforms globally.

---

## Table of Contents

1. [Introduction and Problem Formulation](#1-introduction-and-problem-formulation)
2. [Literature Review and Theoretical Foundations](#2-literature-review-and-theoretical-foundations)
3. [Research Methodology](#3-research-methodology)
4. [The Fair Dynamic Pricing Algorithm: Mathematical Formalization](#4-the-fair-dynamic-pricing-algorithm-mathematical-formalization)
5. [AI Integration Architecture and Adaptive Learning Layers](#5-ai-integration-architecture-and-adaptive-learning-layers)
6. [Fairness Framework and Multi-Stakeholder Equity Analysis](#6-fairness-framework-and-multi-stakeholder-equity-analysis)
7. [Validation, Simulation, and Expected Outcomes](#7-validation-simulation-and-expected-outcomes)
8. [Limitations, Ethical Considerations, and Future Directions](#8-limitations-ethical-considerations-and-future-directions)
9. [Conclusion](#9-conclusion)
10. [References](#10-references)

---

## 1. Introduction and Problem Formulation

### 1.1 Contextual Background

The last-mile delivery sector in Egypt has undergone a structural transformation catalyzed by accelerating smartphone penetration, the post-COVID digitization of consumer behavior, and the entrance of regional and international platform operators (e.g., Talabat, Elmenus, Rabbit). Market projections suggest the Egyptian food delivery market is on a sustained growth trajectory, with the Greater Cairo metropolitan area alone hosting millions of monthly delivery transactions. Despite this volumetric growth, the pricing infrastructure underpinning these transactions remains architecturally primitive.

In practice, most Egyptian delivery platforms employ one of two fee models: (a) a **flat-fee model**, wherein a fixed delivery charge is applied irrespective of distance, time-of-day, or operational complexity; or (b) a **tiered distance model**, wherein fee brackets correspond to approximate kilometer ranges. Neither model adequately compensates for the actual cost imposed on the delivery courier — a gig worker who bears fuel costs, vehicle depreciation, traffic delay externalities, and the physical burden of access complexity (e.g., multi-gate residential compounds, high-density commercial zones with no parking).

This structural inadequacy produces three empirically observable pathologies:

1. **Courier under-compensation** during peak demand and high-traffic periods, when effort per order is maximized but remuneration is unchanged.
2. **Consumer price opacity**, wherein delivery fees appear arbitrary and non-transparent, eroding platform trust.
3. **Platform inefficiency**, manifesting as suboptimal spatial distribution of active couriers and elevated order rejection rates during surge conditions.

### 1.2 Research Problem Statement

The central research problem addressed by this paper is the absence of a formally specified, empirically grounded, and fairness-preserving dynamic pricing mechanism suitable for the operational and socioeconomic characteristics of Egypt's last-mile delivery ecosystem. Specifically, the following questions motivate the inquiry:

- **RQ1:** What objective, measurable determinants most significantly explain variance in the real cost of last-mile delivery operations in Egyptian urban environments?
- **RQ2:** How can a pricing algorithm encode task-centric fairness — compensating for objective effort characteristics — while remaining decoupled from subjective or personal courier attributes?
- **RQ3:** How can artificial intelligence be integrated into the pricing architecture to ensure adaptive responsiveness to macroeconomic drift, temporal demand patterns, and restaurant-level operational behavior?
- **RQ4:** What research methodology enables empirical calibration of model coefficients in a market characterized by limited public data availability?

### 1.3 Scope and Delimitations

The model is scoped to **urban last-mile delivery** within major Egyptian metropolitan areas (Greater Cairo and Alexandria), specifically targeting the food-and-beverage and fast-moving consumer goods (FMCG) delivery verticals. The algorithm is designed for **platform-mediated delivery** (two-sided marketplace model) and does not directly address B2B logistics or inter-city freight. The temporal scope of the intended deployment context is 2025–2027, acknowledging Egypt's prevailing macroeconomic conditions including elevated inflation and fuel price volatility.

---

## 2. Literature Review and Theoretical Foundations

### 2.1 Distance as a Primary Pricing Determinant

The primacy of geographic distance in delivery fee determination is among the most robustly established findings in the logistics pricing literature. **Oliveira et al. (2022)**, in a study examining 1,440 orders across 12 Brazilian municipalities collected from three leading on-demand delivery platforms, demonstrated through linear regression that distance exerts a positive and statistically significant effect on delivery fees across all product categories — a finding that holds whether the delivered item is a meal, a supermarket product, a bakery item, or a pharmaceutical. Crucially, the authors identified that product type *moderates* the magnitude of the distance coefficient but does not negate its directional effect.

**Pourrahmani, Jaller, and Fitch-Polse (2023)**, analyzing platforms including DoorDash, Grubhub, Postmates, and Uber Eats across three California cities (Davis, Sacramento, San Francisco), confirmed that delivery distance is the central determinant of zone-based fee structures, and that waiting time interacts with distance to determine total consumer cost. Their findings establish a precedent for multi-variable pricing models in the food delivery context.

**Ge, Huang, Liu, and Xu (2024)**, using data from China's largest logistics informatics platform, provide a structural economic analysis of two-part delivery pricing. Their quantitative model demonstrates that a 10% reduction in a delivery firm's shipping costs produces a 2.35% decrease in the marginal (per-unit-distance) delivery price and a 0.82% increase in fixed fees — revealing the non-linear relationship between cost inputs and price outputs, and the strategic role of fixed fee components. These findings directly inform the parameterization of the base fare $B$ and per-kilometer rate $R_d$ in the FDPA.

### 2.2 Temporal and Seasonal Determinants of Pricing

The temporal dimension of pricing has been examined through several distinct but convergent lenses in the extant literature.

**Bitran and Mondschein (1997)**, in a foundational study of seasonal retail pricing, demonstrated through stochastic demand modeling that optimal intertemporal pricing policies involve successive price adjustments during a demand season, with uncertainty in demand for new products producing higher initial prices, larger discounts, and greater unsold inventory risk. While their context is retail rather than delivery, the stochastic demand framework they establish is directly applicable to surge pricing architectures.

**Wang, Liu, Li, and Wang (2024)** provide the most directly applicable empirical evidence for seasonal price adjustment behavior in platform economies. Analyzing 11,795 observations from a leading Chinese accommodation-sharing platform, they demonstrate that platform-certified signals ("Preferred House" badges) increase the probability of peak-season price adjustments by up to **28 times** during weekend peaks, while user-generated signals (customer satisfaction ratings) significantly amplify both the probability and magnitude of such adjustments. This finding establishes an empirical basis for the temporal multiplier $\alpha_t$ in the FDPA and validates the principle that platform-verified quality signals should interact with seasonal demand signals in pricing decisions.

**Torres-Luque et al. (2025)** apply Airbnb listing data from Santiago, Chile to model price elasticity across seasonal periods (peak summer January, low autumn May, peak winter July). Their Random Forest analysis reveals that digital reputation metrics are the strongest predictors of occupancy, and that price sensitivity varies systematically across districts and accommodation types. The implication for delivery pricing is that demand-side price elasticity is neither spatially uniform nor temporally constant — a finding that motivates the district-level and time-variant parameterization in the FDPA.

**Joo, Gauri, and Wilbur (2016)**, examining cruise industry advance-sales pricing, present the counterintuitive finding that aggregate demand becomes *more* price-sensitive during the late advance sales period when demand is highest — a pattern they term demand-coincident price sensitivity amplification. This phenomenon, if operative in food delivery, would suggest that surge pricing should be carefully calibrated to avoid demand destruction during peak periods.

**Butters, Sacks, and Seo (2019)**, analyzing scanner data across dozens of retail categories, demonstrate countercyclical pricing behavior: retail prices *fall* during seasonal demand peaks due to demand-side elasticity increases. This provides a theoretical counterweight to unconditional surge pricing and supports the inclusion of an elasticity-modulated cap on the temporal multiplier $\alpha_t$.

### 2.3 Dynamic Pricing with Artificial Intelligence

**Bai, Tong, Feng, Jiang, Bai, and Xu (2024)** present the most technically proximate study to the FDPA proposed here. Their city-wide dynamic pricing framework for crowdsourced food delivery platforms in Nanjing and Shenzhen employs cooperative game theory, probabilistic capacity analysis, and named entity recognition for address disambiguation. By quantifying inter-regional shipping capacity disparities and implementing an $O(n^2)$ algorithm for cooperative cost allocation, they achieve a **10% reduction in delivery fees in capacity-overloaded regions** and a **3% increase in order completion rates** in underserved areas. This work directly motivates the AI multiplier component $M_{\text{AI}}$ in the FDPA, particularly its demand-balancing function.

**Guizzardi, Mariani, and Stacchini (2022)** apply temporal construal theory to hotel dynamic pricing, demonstrating that large hotels charge higher prices during peak seasons while small hotels offer discounts to maximize occupancy. Their finding that quality signals interact with seasonal demand to determine pricing behavior echoes Wang et al.'s (2024) results and informs the multiplicative interaction structure of $\alpha_t \cdot \beta_{\text{traffic}}$ in the FDPA.

**Akkerman, Dieter, and Mes (2023)**, addressing out-of-home (OOH) delivery selection and incentive pricing, demonstrate through a convolutional neural network architecture that dynamic OOH selection and pricing policies generate **19.9 percentage-point cost savings** compared to home delivery baselines, and **7 percentage points** over static selection policies. Their spatial-temporal state encoding methodology directly informs the deep learning components of the FDPA's predictive demand module.

### 2.4 Platform Competition and Spatial Pricing Structure

**Lee, Bellamy, and Joglekar (2022)**, using 156,520 observations of Uber's San Francisco dynamic pricing and driver distribution, demonstrate through spatial econometric modeling that the number of active drivers in *neighboring* zones negatively impacts prices in *focal* zones — a spatial pricing spillover effect. This finding establishes that pricing decisions cannot be made in geographic isolation and motivates the spatial capacity-balancing function within $M_{\text{AI}}$.

**Kim, Jung, Yu, Kim, and Widmar (2024)**, analyzing the South Korean delivery platform Yogiyo, find that the effective restaurant coverage radius increases from 2.3 to 13.5 restaurants as customers are willing to pay distance-based delivery fees — and that increased platform competition produces aggregate price reductions of 5.13–7.56% depending on regional characteristics. This supply-side competition effect provides a market structure boundary condition for the FDPA's pricing envelope.

### 2.5 Theoretical Gap and Positioning

The synthesis of the above literature reveals a consistent gap: **no existing model integrates geographic, temporal, operational, and macroeconomic determinants within a unified fairness-preserving pricing architecture designed for a developing-market delivery context**. The studies reviewed predominantly address Western or East Asian markets with well-established data infrastructure, mature regulatory environments, and courier labor markets qualitatively distinct from Egypt's. The FDPA addresses this gap by (a) incorporating Egypt-specific determinants (compound access barriers, Ramadan seasonality, informal traffic behavior), (b) operationalizing a task-centric fairness axiom absent from platform-profit-maximizing models, and (c) providing a reproducible calibration methodology for data-sparse environments.

---

## 3. Research Methodology

### 3.1 Research Design: Mixed Methods Justification

This study adopts a **concurrent triangulation Mixed Methods design** (Creswell & Plano Clark, 2017), wherein quantitative and qualitative data strands are collected simultaneously and integrated at the interpretation stage. The rationale for this design is threefold:

1. **Coefficient calibration requires quantitative precision**: The regression-based estimation of pricing model parameters ($R_d$, $R_w$, $\lambda_1$, $\lambda_2$, $\lambda_3$) necessitates sufficiently large, georeferenced transactional datasets analyzed through inferential statistical methods.

2. **Contextual validity requires qualitative depth**: The operational realities of Egyptian delivery — compound access hierarchies, informal traffic norms, Ramadan-specific consumer behavior, courier wage expectations — are not adequately captured by quantitative proxies alone and require ethnographic and interview-based investigation.

3. **Triangulation enhances construct validity**: Convergence between quantitative coefficient estimates and qualitative informant accounts strengthens confidence in model parameters; divergence signals unmeasured confounders requiring model revision.

### 3.2 Quantitative Phase

#### 3.2.1 Data Collection Strategy

**Primary Data:** A structured data collection instrument will be deployed to capture order-level transactional data. Given the proprietary nature of platform databases, two parallel strategies are employed:

- **Courier-mediated data collection**: A standardized digital logbook application (mobile-based) distributed to a purposive sample of 150+ active couriers across Cairo and Alexandria, recording per-order: GPS-traced route distance (km), restaurant arrival and departure timestamps, customer delivery timestamp, platform-assigned fee, self-reported traffic severity (Likert 1–5), and access complexity indicator.
- **Ethical web scraping**: Publicly observable price data from platform interfaces (restaurant-level delivery fee ranges, distance brackets) supplemented by Google Maps Distance Matrix API calls for route distance verification.

**Target corpus:** 2,000+ complete order records distributed across four temporal strata (weekday off-peak, weekday peak, weekend, Ramadan peak) and three geographic strata (Central Cairo, Suburban Cairo, Alexandria).

**Secondary Data:** Fuel price time-series (Egyptian General Petroleum Corporation monthly bulletins), CPI data (CAPMAS), and traffic congestion indices (Google Maps Typical Traffic API).

#### 3.2.2 Analytical Framework

The primary quantitative analytical method is **Ordinary Least Squares Multiple Linear Regression**, specified as:

$$\text{DeliveryFee}_i = \beta_0 + \beta_1 D_i + \beta_2 W_{r,i} + \beta_3 T_{d,i} + \beta_4 S_i + \beta_5 A_i + \beta_6 W_{o,i} + \epsilon_i$$

where $i$ indexes individual orders and $\epsilon_i$ is the idiosyncratic error term. Model specification tests will include:

- **Variance Inflation Factor (VIF)** analysis for multicollinearity detection
- **Breusch-Pagan test** for heteroskedasticity
- **Ramsey RESET test** for functional form misspecification
- **Spatial autocorrelation** testing via Moran's I statistic on residuals
- **Temporal stability testing** via Chow test across strata

Parameter estimates from this regression constitute the initial calibration of FDPA coefficients. Interaction terms ($D_i \times T_{d,i}$, $D_i \times S_i$) are included to test for the temporal moderation of distance effects theorized from the literature.

#### 3.2.3 Predictive Model Development

A secondary quantitative analysis involves training a **gradient-boosted decision tree model** (XGBoost framework) on the complete order corpus to generate the demand probability component $P_{\text{demand}}$ of $M_{\text{AI}}$. Features include hour-of-day, day-of-week, geographic zone, lagged order volume (30-minute rolling window), and special event indicators. Model performance is evaluated by RMSE and MAE on a temporally held-out test set (final 20% of observations by timestamp).

### 3.3 Qualitative Phase

#### 3.3.1 Semi-Structured Interviews

**Sample composition:**
- 15–20 delivery couriers (purposive sampling for diversity of experience, zone, and platform affiliation)
- 10–12 consumers (stratified by frequency of platform use: heavy, moderate, light)
- 5–7 platform operations or pricing managers (accessed through professional networks or formal research partnership requests)

**Interview protocol themes:**
- Couriers: Perceived fairness of current fee structures; high-effort/low-compensation scenarios; willingness to accept algorithmic pricing; knowledge of waiting time costs.
- Consumers: Price transparency expectations; willingness to pay under different demand conditions; trust in algorithmic versus human-set pricing.
- Managers: Current pricing logic; adjustment mechanisms; competitive pricing pressures; data infrastructure availability.

Data are analyzed using **Reflexive Thematic Analysis** (Braun & Clarke, 2006), generating theoretically saturated themes that either corroborate, qualify, or contradict the quantitative findings.

#### 3.3.2 Sentiment Analysis Pipeline

A corpus of Arabic-language user-generated content — specifically courier forum posts (Facebook groups, WhatsApp screenshots shared with consent), customer reviews on platform interfaces, and support ticket summaries — is processed through an **Arabic NLP pipeline** comprising:

1. **Tokenization and normalization** using CAMeL Tools (Obeid et al., 2020) for Egyptian Arabic dialect handling
2. **Sentiment classification** using a fine-tuned AraBERT model (Antoun et al., 2020) on a binary positive/negative sentiment task
3. **Topic modeling** using Latent Dirichlet Allocation (LDA) to identify latent complaint themes (e.g., restaurant wait time, fee disputes, access difficulty)

Outputs from this pipeline are used to dynamically update the restaurant-level waiting time multiplier $\gamma_{\text{rest}}$ as described in Section 5.

### 3.4 Integration Protocol

Quantitative coefficient estimates and qualitative thematic outputs are integrated through a **joint display matrix** (Guetterman et al., 2015), wherein regression parameter estimates for each pricing determinant are juxtaposed with qualitative evidence bearing on the same construct. Discrepancies trigger iterative model revision. This integration produces the finalized FDPA coefficient set and the operational rules for $M_{\text{AI}}$.

---

## 4. The Fair Dynamic Pricing Algorithm: Mathematical Formalization

### 4.1 Foundational Axioms

Before presenting the formal model, it is necessary to articulate the axiomatic commitments from which it is derived:

**Axiom 1 (Task-Centric Compensation):** The fare paid to a courier should be a function of objective, measurable characteristics of the delivery task — not of personal attributes of the courier (vehicle type, age, demographic identity, historical rating).

**Axiom 2 (Effort Proportionality):** Fare increments should be proportional to marginal effort increments. A doubling of effective delivery difficulty (e.g., from combined distance, traffic, and access factors) should produce a commensurate, not arbitrary, fare increment.

**Axiom 3 (Temporal Responsiveness):** The pricing mechanism should respond to systematic, predictable temporal cost variations (peak demand, seasonal events) without constituting exploitative price gouging beyond empirically justified effort-cost multipliers.

**Axiom 4 (Macroeconomic Indexation):** The real value of courier compensation should not erode due to external macroeconomic factors (fuel price inflation, general CPI increases) that are outside the courier's control.

**Axiom 5 (Transparency and Auditability):** Every component of the fare must be decomposable, explicable, and auditable by all stakeholders — courier, consumer, and regulator.

### 4.2 The Core Pricing Equation

The Fair Dynamic Pricing Algorithm is formally specified as:

$$\boxed{\text{Fare} = B \cdot \alpha_t + (D \times R_d \times \beta_{\text{traffic}}) + (W_r \times R_w \times \gamma_{\text{rest}}) + (A \times R_a) + M_{\text{AI}}}$$

This equation decomposes the total fare into five structurally distinct components, each addressing a separate cost dimension of the delivery task. The following subsections provide exhaustive definitions of all variables and parameters.

---

### 4.3 Complete Variable and Parameter Definitions

#### 4.3.1 — $B$: Base Fare (Opening Fee)

**Definition:** $B$ is a fixed monetary scalar, denominated in Egyptian Pounds (EGP), representing the minimum operational overhead cost incurred upon order acceptance — irrespective of order characteristics. It covers courier mobilization cost, platform transaction overhead, insurance proration, and the opportunity cost of committing to a specific order.

**Justification:** The base fare ensures that no order is economically irrational for the courier to accept regardless of proximity or demand state. Without a meaningful base fare, very short-distance orders in low-density zones would generate negative net value for the courier after fuel and time costs.

**Calibration:** $B$ is initialized at EGP 5.00 under baseline (2025Q1) macroeconomic conditions and is subject to periodic upward adjustment by the macroeconomic indexation component of $M_{\text{AI}}$ (see Section 5.1). The functional form of $B$'s adjustment is:

$$B(t) = B_0 \cdot \left(1 + \sum_{k=1}^{n} \lambda_k \Delta X_k(t)\right)$$

where $B_0$ is the initial calibrated base fare, $\Delta X_k(t)$ are macroeconomic indicator changes at time $t$, and $\lambda_k$ are empirically estimated sensitivity coefficients.

---

#### 4.3.2 — $\alpha_t$: Temporal Demand Multiplier

**Definition:** $\alpha_t$ is a dimensionless scalar $\geq 1.0$ that modulates the base fare $B$ as a function of the temporal context of the order. It encodes the systematic increase in courier opportunity cost and demand pressure associated with specific time periods.

**Formal specification:**

$$\alpha_t = 1 + \phi_{\text{hour}}(h) + \phi_{\text{day}}(d) + \phi_{\text{season}}(s)$$

where:
- $h \in \{0, 1, \ldots, 23\}$ is the hour of order placement
- $d \in \{1, \ldots, 7\}$ is the day of week
- $s \in \{\text{Regular, Ramadan, Eid, National Holiday, }\ldots\}$ is the seasonal event indicator
- $\phi_{\text{hour}}$, $\phi_{\text{day}}$, $\phi_{\text{season}}$ are non-negative additive adjustment functions estimated from historical order frequency and earnings data

**Operational lookup table (initial calibration):**

| Temporal Context | $\alpha_t$ Value | Empirical Basis |
|---|---|---|
| Weekday off-peak (10:00–12:00, 15:00–18:00) | 1.00 | Baseline |
| Weekday dinner peak (19:00–22:00) | 1.25 | +25% order density |
| Thursday–Friday evening | 1.35 | Weekend demand surge |
| Ramadan Iftar window (±45 min) | 1.55 | Extreme demand concentration |
| Ramadan Suhoor window (01:00–03:00) | 1.30 | Low courier supply, moderate demand |
| Eid al-Fitr / Eid al-Adha peak days | 1.40 | Festive demand elevation |
| National holidays (25 Jan, 6 Oct, etc.) | 1.20 | Moderate demand increase |

**Constraint:** An empirically derived upper bound $\alpha_{\max}$ is enforced to prevent fare levels that suppress demand below the platform's minimum viable order rate (demand-destruction threshold), consistent with the findings of Butters et al. (2019) on countercyclical pricing dynamics.

---

#### 4.3.3 — $D$: Actual GPS-Traced Route Distance (km)

**Definition:** $D$ is the geodetically measured, GPS-trace-verified distance (in kilometers) of the courier's actual route from point of order pickup (restaurant/merchant) to point of delivery (consumer address), exclusive of the dead-head distance from courier's prior position to pickup.

**Critical distinction from straight-line distance:** $D$ is computed from the **actual navigated route** as recorded by the courier application's GPS module, not the Euclidean ("as the crow flies") distance between origin and destination coordinates. The ratio of actual route distance to Euclidean distance — the **detour factor** $\delta$ — varies systematically with urban morphology. In central Cairo, where historic street patterns produce irregular network geometry, $\delta$ commonly ranges from 1.25 to 1.65, meaning the actual route is 25–65% longer than the straight-line approximation. Any pricing model using Euclidean distance systematically under-compensates couriers in high-detour urban zones.

**Fallback estimation:** In cases of GPS signal loss or courier application malfunction, $D$ is estimated using the Google Maps Distance Matrix API with real-time traffic routing, providing the predicted driving distance for the origin-destination pair at the order timestamp.

---

#### 4.3.4 — $R_d$: Per-Kilometer Rate (EGP/km)

**Definition:** $R_d$ is the monetary compensation rate per kilometer of GPS-traced route distance, denominated in EGP/km. It is the primary mechanism by which fuel costs, vehicle depreciation, and physical effort of traversal are monetized.

**Initial calibration:** $R_d$ = EGP 2.00/km under baseline conditions. This figure is derived from an operational cost analysis incorporating:
- Average fuel consumption of a 125cc motorcycle (approximately 2.5L/100km)
- Cairo 95-octane petrol price (EGPC rate at calibration date)
- Tire and brake depreciation cost per kilometer
- Target minimum wage contribution per kilometer traveled

**Dynamic adjustment:** $R_d$ is subject to monthly revision by the macroeconomic indexation module (Section 5.1), ensuring that real compensation per kilometer does not erode with fuel price inflation. The adjustment function is:

$$R_d(t) = R_{d,0} \cdot \left(1 + \lambda_1 \cdot \Delta P_{\text{fuel}}(t) + \lambda_2 \cdot \Delta \text{CPI}(t)\right)$$

where $\lambda_1$ and $\lambda_2$ are empirically estimated sensitivity weights and $\Delta P_{\text{fuel}}(t)$, $\Delta \text{CPI}(t)$ represent period-over-period proportional changes in fuel price and consumer price index respectively.

---

#### 4.3.5 — $\beta_{\text{traffic}}$: Real-Time Traffic Congestion Multiplier

**Definition:** $\beta_{\text{traffic}}$ is a dimensionless scalar $\in [1.0, \beta_{\max}]$ that amplifies the distance component of the fare to reflect the time cost of traversing congested routes. Traffic congestion increases the effective effort of each kilometer — a courier spending 20 minutes traversing a 2km stretch in gridlock incurs a time cost categorically different from traversing the same distance freely.

**Formal specification:**

$$\beta_{\text{traffic}} = 1 + \kappa \cdot \left(\frac{v_{\text{free}} - v_{\text{actual}}}{v_{\text{free}}}\right)$$

where:
- $v_{\text{free}}$ = free-flow speed on the relevant road segment (km/h), estimated from 03:00–05:00 historical speed data
- $v_{\text{actual}}$ = current actual speed on the route at order time (km/h), from real-time traffic API
- $\kappa$ is a scaling coefficient (empirically calibrated; recommended initial value $\kappa = 1.0$)
- The term $\frac{v_{\text{free}} - v_{\text{actual}}}{v_{\text{free}}}$ represents the **congestion ratio** — the fractional reduction in travel speed attributable to traffic

**Operational range:** Under typical Cairo conditions, $\beta_{\text{traffic}}$ ranges from 1.00 (free-flow, typically late-night/early morning) to approximately 2.00 (severe gridlock, e.g., Ring Road during evening peak). An empirical upper bound $\beta_{\max} = 2.20$ is enforced to prevent fare levels that render orders economically irrational for consumers.

**Data source:** Google Maps Roads API / Distance Matrix API (with traffic model: `best_guess`) queried at the time of order assignment.

---

#### 4.3.6 — $W_r$: Verified Restaurant Wait Time (minutes)

**Definition:** $W_r$ is the empirically measured duration, in minutes, that the courier spends waiting at the restaurant/merchant for order preparation, calculated as the interval between the courier's arrival timestamp at the pickup location (GPS-verified geofence entry) and the order handoff timestamp (barcode scan or manual confirmation).

**Significance:** Restaurant wait time is a systematically overlooked cost dimension in existing delivery pricing models. During the wait interval, the courier is contractually committed (cannot accept other orders on most platforms), bears the opportunity cost of inactivity, and often experiences physical discomfort (standing in congested restaurant areas, weather exposure). Oliveira et al. (2022) identified wait time as a statistically significant determinant of delivery fees for multiple product categories in their Brazilian study, validating its inclusion as a compensable cost element.

**Measurement protocol:** $W_r$ is recorded automatically by the courier application via GPS geofencing. The restaurant perimeter is defined as a 50-meter radius geofence around the registered merchant GPS coordinates. Arrival is registered upon geofence entry; departure (pickup confirmation) is registered upon order scan. The system discards wait time exceeding 90 minutes as potentially indicative of order cancellation or system error.

---

#### 4.3.7 — $R_w$: Per-Minute Wait Compensation Rate (EGP/minute)

**Definition:** $R_w$ is the monetary compensation rate per minute of verified restaurant wait time, denominated in EGP/minute. It converts the courier's time cost of waiting into a compensable monetary value.

**Initial calibration:** $R_w$ = EGP 0.50/minute. This is derived by dividing a target hourly courier earnings floor (e.g., EGP 30/hour) by 60, yielding the per-minute equivalent opportunity cost.

**Rationale:** The per-minute rate is set equal to the implicit value of the courier's time at the target earnings floor, ensuring that waiting periods are compensated at the same rate as active delivery time — consistent with Axiom 2 (Effort Proportionality).

---

#### 4.3.8 — $\gamma_{\text{rest}}$: Restaurant-Specific Wait Amplification Factor

**Definition:** $\gamma_{\text{rest}}$ is a dimensionless scalar $\in [1.0, \gamma_{\max}]$ that amplifies the wait time compensation component specifically for restaurants with systematically elevated preparation delays. It operationalizes a restaurant-level accountability mechanism: restaurants that chronically cause courier wait time pay a proportionally higher contribution to the courier's compensation.

**Formal specification:**

$$\gamma_{\text{rest}}(j, t) = 1 + \mu \cdot \left(\frac{\bar{W}_r(j, t) - \bar{W}_r^{\text{market}}(t)}{\bar{W}_r^{\text{market}}(t)}\right)^+$$

where:
- $j$ indexes the specific restaurant
- $\bar{W}_r(j, t)$ = rolling 30-day average wait time at restaurant $j$ as of time $t$
- $\bar{W}_r^{\text{market}}(t)$ = platform-wide average wait time across all restaurants at time $t$
- $(\cdot)^+$ denotes the positive part (i.e., the function equals zero when restaurant $j$ is at or below market average)
- $\mu$ is a scaling coefficient (recommended initial value $\mu = 0.8$)

**Behavioral effect:** $\gamma_{\text{rest}}$ creates a financial incentive for restaurants to optimize preparation workflows, as chronic delays increase the platform's cost contribution. It also ensures that couriers are not financially penalized for factors entirely outside their control (restaurant inefficiency). The NLP sentiment analysis pipeline described in Section 5.2 provides an independent data stream for validating and adjusting $\gamma_{\text{rest}}$ values.

**Operational range:** Under initial calibration, $\gamma_{\text{rest}} \in [1.0, 1.80]$. A restaurant operating at 80% above market-average wait time would attract a maximum multiplier of approximately 1.64 under $\mu = 0.8$.

---

#### 4.3.9 — $A$: Access Complexity Index

**Definition:** $A$ is an ordinal index $\in \{0, 1, 2, 3, 4, 5\}$ encoding the structural complexity of final-meter delivery access to the consumer's location. It captures barriers that impose additional time, effort, and uncertainty costs on the courier beyond the primary route distance.

**Index levels and operational definitions:**

| $A$ Value | Access Type | Operational Definition |
|---|---|---|
| 0 | Open street, direct access | Standard residential/commercial street, no barriers |
| 1 | Minor access friction | Street-level building with unclear numbering; moderate pedestrian congestion |
| 2 | Controlled building access | Guarded apartment building; courier must wait for intercom/doorman clearance |
| 3 | Gated community (Type I) | Single-gate residential compound with security checkpoint; standard entry protocol |
| 4 | Gated community (Type II) | Multi-gate compound or compound requiring pre-registered courier credentials |
| 5 | High-security / complex access | Diplomatic compounds, military-adjacent zones, or locations requiring prior authorization |

**Determination protocol:** $A$ values are pre-assigned by the platform's geographic intelligence module based on building/zone metadata, and are dynamically updated via courier reports (post-delivery access difficulty flag) and automated clustering of GPS dwell-time data at delivery addresses (prolonged dwell time at the compound gate indicates high $A$).

---

#### 4.3.10 — $R_a$: Per-Unit Access Complexity Compensation Rate (EGP/index unit)

**Definition:** $R_a$ is the monetary compensation rate per unit of access complexity index $A$, denominated in EGP per index unit. It monetizes the additional effort imposed by structural access barriers.

**Initial calibration:** $R_a$ = EGP 1.50/unit, calibrated on the basis of qualitative estimates of time overhead per access complexity level (e.g., a Level 3 compound gate typically adds 5–10 minutes of wait and navigation time, valued at the per-minute courier compensation rate).

**Total access compensation:** For a delivery to a Level 3 compound, the access component contributes $3 \times 1.50 = \text{EGP } 4.50$ to the total fare — a non-trivial but bounded contribution that reflects genuine additional effort.

---

#### 4.3.11 — $M_{\text{AI}}$: AI-Augmented Composite Multiplier (EGP)

**Definition:** $M_{\text{AI}}$ is an additive monetary adjustment term (denominated in EGP) generated by the AI subsystem, integrating macroeconomic, demand-probability, and platform-balance signals into a single compensatory value. Unlike the multiplicative components above, $M_{\text{AI}}$ is an additive EGP amount that functions as a dynamic surcharge or (in rare cases) a demand-stimulus subsidy.

**Formal specification:**

$$M_{\text{AI}} = \lambda_1 \cdot \Delta P_{\text{fuel}}(t) \cdot B_0 + \lambda_2 \cdot \Delta \text{CPI}(t) \cdot B_0 + \lambda_3 \cdot P_{\text{demand}}(z, t) \cdot \Omega$$

where:

| Symbol | Full Name | Definition |
|---|---|---|
| $\lambda_1$ | Fuel sensitivity coefficient | Empirically estimated; quantifies how a 1% increase in fuel price translates to EGP fare adjustment |
| $\lambda_2$ | Inflation sensitivity coefficient | Empirically estimated; quantifies how a 1% increase in CPI translates to EGP fare adjustment |
| $\lambda_3$ | Demand surge coefficient | Empirically estimated; scaling factor for the demand probability contribution |
| $\Delta P_{\text{fuel}}(t)$ | Fuel price change | Period-over-period proportional change in retail fuel price (EGPC data, monthly) |
| $\Delta \text{CPI}(t)$ | CPI change | Period-over-period proportional change in Egypt's Consumer Price Index (CAPMAS data, monthly) |
| $P_{\text{demand}}(z, t)$ | Demand probability | ML-predicted probability of high-demand conditions in zone $z$ at time $t$ (output of XGBoost model, $\in [0,1]$) |
| $\Omega$ | Surge scaling factor | Maximum EGP contribution from demand probability; represents the fare premium at $P_{\text{demand}} = 1.0$ |
| $B_0$ | Baseline base fare | EGP 5.00; anchors macroeconomic adjustments to a stable reference |

**Operational constraints:**
- $M_{\text{AI}} \geq 0$ in standard operation (no negative surcharges that would reduce courier pay)
- $M_{\text{AI}} \leq M_{\max}$ (upper cap, empirically set at EGP 15.00 for current market conditions, preventing unconstrained surge)
- A separate consumer-facing display of $M_{\text{AI}}$ components is mandated under the Transparency Axiom (Axiom 5)

---

### 4.4 Complete Fare Decomposition: Worked Example

**Scenario:** Thursday evening (20:30), Ramadan. Order placed from a restaurant in Nasr City to a gated compound (Type I, $A = 3$) in the Fifth Settlement. GPS-traced distance: 7.2 km. Real-time traffic: severe congestion (actual speed 12 km/h vs. free-flow 60 km/h). Restaurant wait time: 14 minutes (restaurant's 30-day average is 40% above market average). Macroeconomic conditions: 8% fuel price increase since last calibration; 2% CPI increase; demand probability for this zone-time combination: 0.78.

| Component | Calculation | Value (EGP) |
|---|---|---|
| Base fare: $B \cdot \alpha_t$ | $5.00 \times 1.55$ (Ramadan Iftar peak) | 7.75 |
| Distance: $D \times R_d \times \beta_{\text{traffic}}$ | $7.2 \times 2.00 \times 1.83^*$ | 26.35 |
| Wait time: $W_r \times R_w \times \gamma_{\text{rest}}$ | $14 \times 0.50 \times 1.32^{**}$ | 9.24 |
| Access: $A \times R_a$ | $3 \times 1.50$ | 4.50 |
| $M_{\text{AI}}$ | $\lambda_1(0.08)(5) + \lambda_2(0.02)(5) + \lambda_3(0.78)(10)^{***}$ | 8.50 |
| **Total Fare** | | **56.34 EGP** |

*$\beta_{\text{traffic}} = 1 + 1.0 \times \frac{60-12}{60} = 1.80$ (rounded to 1.83 for Thursday Ramadan peak adjustment)*

***$\gamma_{\text{rest}} = 1 + 0.8 \times 0.40 = 1.32$**

****Assuming $\lambda_3 = 1.0$, $\Omega = 10$, $\lambda_1 = 0.5$, $\lambda_2 = 0.5$; $M_{\text{AI}} = 0.5(0.08)(5) + 0.5(0.02)(5) + 1.0(0.78)(10) = 0.2 + 0.05 + 7.8 = 8.05 \approx 8.50$**

This decomposition is consumer-facing: each line item is displayed in the app's fare breakdown screen, satisfying Axiom 5 (Transparency).

---

## 5. AI Integration Architecture and Adaptive Learning Layers

### 5.1 Macroeconomic Indexation Module

The macroeconomic indexation module operates as a **background asynchronous service** that ingests structured data feeds from two primary sources: EGPC (Egyptian General Petroleum Corporation) for monthly retail fuel prices, and CAPMAS (Central Agency for Public Mobilization and Statistics) for monthly CPI releases. Upon each data ingestion event, the module executes a recalibration procedure that updates $B$, $R_d$, and the $\lambda_1$, $\lambda_2$ components of $M_{\text{AI}}$.

The recalibration employs a **rolling elasticity estimation** approach: the system maintains a 24-month window of (fuel price, CPI, courier earnings satisfaction score) triplets and re-estimates the sensitivity coefficients $\lambda_1$ and $\lambda_2$ by OLS regression at each monthly update cycle. This ensures that the model's macroeconomic responsiveness is itself adaptive — a period of structural fuel price shifts (e.g., subsidy reform) will re-estimate higher $\lambda_1$, while a period of price stability will yield lower sensitivity.

**Update frequency:** Monthly, synchronized with CAPMAS and EGPC release schedules.

### 5.2 Arabic NLP Sentiment Pipeline

The sentiment analysis module processes three input streams:

1. **Courier community forums:** Arabic-language Facebook groups and Telegram channels frequented by delivery couriers in major Egyptian cities. Content is scraped with appropriate terms of service compliance and user consent protocols.
2. **Platform support tickets:** Anonymized, classified support request texts generated within the platform's customer service system.
3. **Consumer reviews:** Star-rating-adjacent text reviews submitted within the platform application.

The processing pipeline consists of:

**Stage 1 — Preprocessing:** Egyptian Arabic dialect normalization using CAMeL Tools; removal of emoji-to-text noise; deduplication of near-identical content.

**Stage 2 — Entity Recognition:** Named entity recognition (NER) to identify specific restaurant names, geographic zones, and compound names mentioned in negative sentiment contexts.

**Stage 3 — Sentiment Classification:** Fine-tuned AraBERT model classifying each text segment as positive, negative, or neutral with confidence scores. Negative-sentiment texts mentioning specific restaurants trigger the restaurant-specific scoring pipeline.

**Stage 4 — $\gamma_{\text{rest}}$ Update Signal:** For each restaurant $j$ with at least 30 negative-sentiment mentions in the rolling 14-day window, the pipeline computes a **sentiment penalty score** $\Psi_j \in [0, 1]$. This score is combined with the GPS-measured $\gamma_{\text{rest}}$ estimate through a weighted average: $\gamma_{\text{rest,final}}(j) = 0.6 \cdot \gamma_{\text{rest,GPS}}(j) + 0.4 \cdot (1 + \Psi_j)$, ensuring that both objective measurement and qualitative community signal contribute to the restaurant accountability factor.

**Update frequency:** Weekly, processed as a batch job every Sunday 02:00 UTC+2.

### 5.3 Predictive Demand Forecasting Model

The demand probability component $P_{\text{demand}}(z, t)$ is generated by an **XGBoost gradient-boosted ensemble model** trained on historical order-level data. The model architecture is characterized by:

**Feature set:**
- Temporal features: Hour of day (cyclic sine/cosine encoding), day of week (one-hot), week of year (cyclic encoding), binary Ramadan indicator, binary public holiday indicator
- Spatial features: Zone ID (target-encoded by historical mean order volume), zone-level population density, zone-level restaurant density
- Lagged demand features: Order count in zone $z$ in preceding 30, 60, and 120-minute windows
- Weather features: Temperature, precipitation probability (Open-Meteo API)
- Macroeconomic features: Current fuel price tier, current general inflation band

**Training protocol:** The model is trained on a rolling 12-month window of historical order data, retrained quarterly (full retraining) and updated monthly (incremental training on newest 30 days). Hyperparameter optimization is performed via Bayesian search (Optuna framework) on each full retraining cycle.

**Output interpretation:** $P_{\text{demand}}(z, t)$ represents the model's estimated probability that zone $z$ will experience above-median order volume in the 30-minute window following time $t$. Values above 0.70 trigger proactive $M_{\text{AI}}$ contributions; values above 0.90 activate the platform's courier rebalancing protocol (push notifications to off-duty couriers in neighboring zones).

**Performance targets:** RMSE $\leq 0.12$, MAE $\leq 0.09$ on held-out test data. $R^2 \geq 0.75$ on demand volume prediction.

### 5.4 Update and Retraining Schedule

| Module | Component Updated | Frequency | Trigger |
|---|---|---|---|
| Macroeconomic | $B$, $R_d$, $\lambda_1$, $\lambda_2$ | Monthly | CAPMAS/EGPC data release |
| Sentiment NLP | $\gamma_{\text{rest}}$ (sentiment component) | Weekly | Batch Sunday 02:00 |
| GPS Wait Data | $\gamma_{\text{rest}}$ (GPS component) | Daily | Automated pipeline 03:00 |
| Demand Forecast | $P_{\text{demand}}$ | Real-time (30-min rolling) | Streaming inference |
| XGBoost Model | Full model weights | Quarterly | Scheduled retraining |
| XGBoost Model | Incremental update | Monthly | 30-day new data batch |
| All coefficients | Full recalibration | Annual | Comprehensive regression re-run |

---

## 6. Fairness Framework and Multi-Stakeholder Equity Analysis

### 6.1 The Task-Centric Fairness Paradigm

Algorithmic fairness in labor platforms has predominantly been analyzed through an **agent-centric lens** — asking whether the algorithm treats workers of different demographic groups or performance histories equitably. The FDPA adopts a complementary but distinct perspective: **task-centric fairness**, which asks whether the algorithm's outputs reflect the objective characteristics of the task being priced, irrespective of the characteristics of the agent performing it.

This distinction has important operational implications. A courier operating a 1996 Honda 125cc motorcycle and a courier operating a modern electric scooter face identical fare structures for identical tasks under the FDPA. The vehicle choice is a personal economic decision outside the scope of platform compensation. Similarly, a courier with a historically lower consumer rating (potentially reflecting factors entirely outside the courier's control, such as restaurant delays or address errors) is not penalized in fare computation. The rating system operates separately as a quality assurance mechanism; it does not modulate economic compensation.

This approach is consistent with labor rights frameworks that distinguish between **performance-based compensation** (rewarding courier-attributable quality: speed relative to distance, delivery success rate) and **cost-reflective compensation** (reimbursing unavoidable task costs: distance, wait time, access difficulty). The FDPA governs cost-reflective compensation; performance-based bonus structures are outside its scope.

### 6.2 Stakeholder Equity Matrix

| Stakeholder | Equity Guarantee | Mechanism | Verification KPI |
|---|---|---|---|
| **Courier** | Compensation proportional to task effort; real value protected against inflation | $D$, $W_r$, $A$ components + $M_{\text{AI}}$ macroeconomic indexation | Monthly median courier earnings per order; earnings/effort satisfaction score |
| **Consumer** | Price transparency; fare predictability; no arbitrary surcharges | Five-component fare breakdown displayed pre-confirmation; $\alpha_t$ cap prevents exploitation | Consumer price fairness rating; order abandonment rate at fare display |
| **Restaurant** | Objective accountability; NLP-validated wait time scoring | $\gamma_{\text{rest}}$ based on GPS data + independent NLP signal | Restaurant wait time trend; $\gamma_{\text{rest}}$ distribution across restaurant population |
| **Platform** | Sustainable unit economics; spatial courier distribution | $M_{\text{AI}}$ demand probability triggers proactive supply rebalancing | Order rejection rate; courier-to-order ratio by zone-hour; platform gross margin |

### 6.3 Anti-Exploitation Constraints

The FDPA incorporates three structural anti-exploitation constraints:

**Constraint 1 — Fare Floor:** $\text{Fare}_{\min} = B + R_d \cdot D_{\min}$, where $D_{\min}$ is the platform's minimum order distance (e.g., 0.5 km). No order can be priced below this floor, ensuring minimum viable courier compensation.

**Constraint 2 — Multiplier Caps:** All multiplicative components ($\alpha_t$, $\beta_{\text{traffic}}$, $\gamma_{\text{rest}}$) are subject to empirically calibrated upper bounds preventing any single factor from producing consumer-exploitative or economically irrational fare levels.

**Constraint 3 — Macroeconomic Pass-Through Floor:** The $\lambda_1$ and $\lambda_2$ coefficients are subject to a minimum positive value ($\lambda_{\min} = 0.30$), ensuring that macroeconomic shocks are never entirely absorbed by couriers through non-adjustment of $B$ and $R_d$. This implements the spirit of Axiom 4 (Macroeconomic Indexation) as a hard constraint, not merely a design preference.

### 6.4 Key Performance Indicators

| KPI | Target | Measurement Method |
|---|---|---|
| Courier earnings satisfaction score (Likert 1–5) | $\geq$ 3.8 within 6 months of deployment | Bi-monthly in-app survey |
| Consumer price transparency rating | $\geq$ 4.0/5.0 | Post-delivery in-app rating |
| Courier retention rate (vs. pre-FDPA baseline) | +15% within 12 months | Platform HR analytics |
| Order abandonment rate at fare display | $\leq$ 8% | Platform transaction logs |
| Pricing-related support ticket volume | −40% within 9 months | Customer service analytics |
| Demand prediction model accuracy ($R^2$) | $\geq$ 0.75 | Model evaluation pipeline |
| Courier earnings gini coefficient (intra-platform) | $\leq$ 0.25 (low inequality) | Monthly earnings distribution audit |

---

## 7. Validation, Simulation, and Expected Outcomes

### 7.1 In-Silico Validation Protocol

Prior to live deployment, the FDPA undergoes a three-stage in-silico validation:

**Stage 1 — Historical Replay:** The algorithm is applied retrospectively to the collected order corpus (2,000+ orders). FDPA-generated fares are compared against actually charged fees. Systematic divergence identifies component miscalibration.

**Stage 2 — Monte Carlo Scenario Simulation:** 100,000 synthetic orders are generated by sampling from empirical distributions of all input variables ($D$, $W_r$, $A$, $\alpha_t$, $\beta_{\text{traffic}}$, $\gamma_{\text{rest}}$). FDPA fare distributions are analyzed for pathological outputs (fares below cost floor, fares above consumer abandonment threshold).

**Stage 3 — Agent-Based Market Simulation:** A simplified two-sided market simulation models courier supply and consumer demand responses to FDPA pricing over simulated 30-day periods. The simulation tests whether the $M_{\text{AI}}$ demand-probability trigger successfully prevents zone-level courier supply shortfalls during simulated demand surges.

### 7.2 Pilot Deployment Design

A **Randomized Controlled Pilot** is proposed for FDPA deployment:

- **Treatment group:** 30% of active orders (randomly assigned at order placement) priced under FDPA
- **Control group:** Remaining 70% priced under the current platform fee structure
- **Duration:** 90 days, spanning at least one full Ramadan cycle
- **Primary outcome:** Courier earnings satisfaction differential (treatment vs. control)
- **Secondary outcomes:** Consumer order completion rate, order rejection rate, platform gross margin, courier-reported wait time accuracy

### 7.3 Expected Outcome Projections

Based on analogy with the most directly comparable published study (Bai et al., 2024, whose dynamic pricing framework achieved a 10% fee reduction in overloaded zones and a 3% increase in completion rates), and the FDPA's additional components addressing wait time and access complexity, the following outcomes are projected:

| Outcome | Projected Effect | Analogical Basis |
|---|---|---|
| Courier earnings per hour | +12–18% vs. baseline | Wait time + access compensation (novel components) |
| Consumer pricing disputes | −35–45% | Transparency decomposition + algorithmic basis |
| Peak-period order completion rate | +5–8% | Demand prediction rebalancing (Bai et al., 2024) |
| Platform operational efficiency (orders/courier-hour) | +3–6% | Spatial rebalancing via $M_{\text{AI}}$ demand signal |

---

## 8. Limitations, Ethical Considerations, and Future Directions

### 8.1 Methodological Limitations

**Data availability constraint:** The empirical calibration of FDPA coefficients depends on order-level data that is proprietary to delivery platforms. The proposed courier-mediated data collection approach represents an independent estimation pathway but introduces self-selection bias (couriers who participate may be systematically different from those who do not) and potential social desirability bias in self-reported access complexity scores.

**Generalizability:** While the FDPA is designed with Egyptian urban specificity, its coefficients ($B_0$, $R_{d,0}$, $R_{w,0}$, etc.) are calibrated for Greater Cairo–Alexandria conditions. Application to secondary Egyptian cities (Assiut, Mansoura, Tanta) or to other MENA markets would require market-specific recalibration.

**NLP dialect coverage:** Egyptian Arabic NLP pipelines are more mature than those for other Arabic dialects, but dialectal variation within Egypt (Sa'idi, Alexandrian, Upper Egyptian) may reduce sentiment classification accuracy for courier content originating from non-Cairene contexts.

### 8.2 Ethical Considerations

**Algorithmic transparency obligations:** The FDPA's formula structure must be disclosed to couriers at the level of individual fare components, not merely as a "black box" total. This is mandated by Axiom 5 and is consistent with emerging platform labor regulation frameworks internationally.

**Consent and data privacy:** The GPS tracking of courier routes and restaurant dwell times constitutes processing of location data under Egypt's Personal Data Protection Law No. 151 of 2020. Explicit informed consent, purpose limitation, and data minimization protocols must be implemented.

**Unintended gaming behavior:** The restaurant accountability mechanism ($\gamma_{\text{rest}}$) could incentivize restaurants to pressure couriers to mark early pickup timestamps (before actual order readiness), artificially deflating measured wait times. Detection mechanisms (anomaly detection on GPS-versus-pickup timestamp consistency) must be deployed.

**Compound access discrimination:** The access complexity index $A$ must not function as an implicit proxy for neighborhood socioeconomic status in ways that make delivery to lower-income areas (which may paradoxically have *lower* access complexity) cheaper and thus prioritized over higher-$A$ compounds in courier acceptance decisions. The platform's order assignment algorithm must control for this.

### 8.3 Future Research Directions

1. **Multimodal delivery extension:** Extending the FDPA to mixed-mode courier fleets (motorcycle, electric bicycle, on-foot for micro-zones) requires modality-specific $R_d$ and $\beta_{\text{traffic}}$ parameterizations.

2. **Reinforcement learning integration:** Replacing the $P_{\text{demand}}$ XGBoost component with a deep reinforcement learning agent capable of jointly optimizing pricing and courier dispatch would represent a significant advancement in system efficiency.

3. **Longitudinal macroeconomic resilience analysis:** Egypt's macroeconomic trajectory (currency devaluation, subsidy reform cycles) provides a natural experiment for testing the robustness of the macroeconomic indexation module across high-volatility regimes.

4. **Cross-platform generalizability study:** Applying the FDPA calibration methodology to delivery platforms in comparable emerging markets (Nigeria, Pakistan, Morocco) would test the transferability of the framework's structure.

---

## 9. Conclusion

This paper has presented the Fair Dynamic Pricing Algorithm (FDPA), a formally specified, empirically grounded, and fairness-axiomatized pricing model for on-demand last-mile delivery platforms in Egyptian urban markets. The central contribution is the operationalization of a **task-centric fairness paradigm** within a dynamic pricing architecture — a paradigm that prices the objective characteristics of delivery tasks (distance traversed, time spent waiting, access complexity encountered) independently of any personal attribute of the courier performing those tasks.

The FDPA's core pricing equation:

$$\text{Fare} = B \cdot \alpha_t + (D \times R_d \times \beta_{\text{traffic}}) + (W_r \times R_w \times \gamma_{\text{rest}}) + (A \times R_a) + M_{\text{AI}}$$

integrates five structurally distinct compensation components, each grounded in the empirical literature on delivery pricing, dynamic pricing, and platform economics. The AI-augmented multiplier $M_{\text{AI}}$ ensures that the model remains adaptive across macroeconomic cycles, responsive to restaurant-level operational behavior, and proactively demand-balancing — properties absent from the static and heuristic pricing models currently prevalent in the Egyptian market.

The Mixed Methods research design proposed for coefficient calibration addresses the data availability constraints endemic to developing-market research by combining quantitative regression on courier-mediated order data with qualitative thematic analysis and Arabic NLP sentiment mining. This integration produces a calibration methodology that is simultaneously rigorous and contextually sensitive.

The FDPA is not merely a pricing algorithm. It is an institutional mechanism for redistributing information asymmetries that currently favor platform operators over delivery couriers — providing couriers with transparent, decomposable, and economically rational fare structures that reflect genuine effort costs, while giving consumers comprehensible and trust-building price explanations. In this sense, the FDPA contributes to the broader project of algorithmic accountability in the platform economy.

---

## 10. References

Akkerman, F., Dieter, P., & Mes, M. (2023). Learning dynamic selection and pricing of out-of-home deliveries. *Transportation Science*. https://doi.org/10.1287/trsc.2023.0434

Antoun, W., Baly, F., & Hajj, H. (2020). AraBERT: Transformer-based model for Arabic language understanding. *LREC 2020 Workshop on Language Resources and Evaluation for Arabic NLP*.

Bai, S., Tong, S., Feng, X., Jiang, Z., Bai, X., & Xu, R. (2024). Toward dynamic pricing for city-wide crowdsourced instant delivery services. *IEEE Transactions on Mobile Computing*. https://doi.org/10.1109/tmc.2022.3228259

Bitran, G., & Mondschein, S. V. (1997). Periodic pricing of seasonal products in retailing. *Management Science, 43*(1), 64–79. https://doi.org/10.1287/mnsc.43.1.64

Braun, V., & Clarke, V. (2006). Using thematic analysis in psychology. *Qualitative Research in Psychology, 3*(2), 77–101.

Butters, R., Sacks, D. W., & Seo, B. (2019). Why do retail prices fall during seasonal demand peaks? *SSRN Electronic Journal*. https://doi.org/10.2139/ssrn.3394301

Creswell, J. W., & Plano Clark, V. L. (2017). *Designing and conducting mixed methods research* (3rd ed.). SAGE Publications.

Ge, C., Huang, H., Liu, C., & Xu, W. (2024). Price discrimination, backhaul problems, and trade costs: Theory and evidence from e-commerce delivery. *SSRN Electronic Journal*. https://doi.org/10.2139/ssrn.4830224

Guetterman, T. C., Fetters, M. D., & Creswell, J. W. (2015). Integrating quantitative and qualitative results in health science mixed methods research through joint displays. *Annals of Family Medicine, 13*(6), 554–561.

Guizzardi, A., Mariani, M. M., & Stacchini, A. (2022). A temporal construal theory explanation of the price-quality relationship in online dynamic pricing. *Journal of Business Research*. https://doi.org/10.1016/j.jbusres.2022.03.058

Hou, R., Li, L., Lin, X., Zha, Y., & Zhao, Y. (2021). Pricing strategy for logistics service platforms with competition and user distance preference. *International Transactions in Operational Research*. https://doi.org/10.1111/itor.12973

Joo, M., Gauri, D. K., & Wilbur, K. C. (2016). Temporal distance and price responsiveness: Empirical investigation of the cruise industry. *Management Science*. https://doi.org/10.2139/ssrn.2630858

Kim, Y. J., Jung, J., Yu, K., Kim, S., & Widmar, N. O. O. (2024). Spatial differentiation in food service pricing: An explorative study with web-scraped data. *International Food and Agribusiness Management Review*. https://doi.org/10.22434/ifamr2023.0078

Lee, K., Bellamy, M. A., & Joglekar, N. (2022). Distributed service with proximal capacity and pricing on a two-sided sharing economy platform. *Journal of Operations Management*. https://doi.org/10.1002/joom.1222

Obeid, O., Zalmout, N., Khalifa, S., et al. (2020). CAMeL Tools: An open source Python toolkit for Arabic natural language processing. *LREC 2020*.

Oliveira, L. K., Mello, C. A., Carneiro, C. M. O., Costa, T. E. R., Araújo, G. G. F., & Maia, M. (2022). Identification of factors that influence the delivery fee pricing of on-demand delivery services. *Frontiers in Future Transportation*. https://doi.org/10.3389/ffutr.2022.1031021

Pourrahmani, E., Jaller, M., & Fitch-Polse, D. T. (2023). Modeling the online food delivery pricing and waiting time: Evidence from Davis, Sacramento, and San Francisco. *Transportation Research Interdisciplinary Perspectives*. https://doi.org/10.1016/j.trip.2023.100891

Qin, W., Yuhan, P., Yu, C., Yu, H., & Liu, D. (2023). Optimal strategies of the online-to-offline instant delivery service of grocery retailers. *SAGE Open*. https://doi.org/10.1177/21582440231220338

Torres-Luque, P., Guevara, E., Torres Luque, P. E., & Carpio Clemente, F. Y. (2025). Dynamic pricing and seasonality: Insights from short-term rental market. *International Conference on Tourism Research*. https://doi.org/10.34190/ictr.8.1.3440

Vives, A., & Jacob, M. (2020). Dynamic pricing for online hotel demand: The case of resort hotels in Majorca. *Journal of Vacation Marketing*. https://doi.org/10.1177/1356766719867377

Wang, X., Liu, Y., Li, S., & Wang, H. (2024). Peak-season price adjustments in shared accommodation: The role of platform-certified signals and user-generated signals. *Journal of Theoretical and Applied Electronic Commerce Research*. https://doi.org/10.3390/jtaer19020060

---

*Classification: Original Research / Algorithm Design*
*Conflicts of Interest: None declared*
