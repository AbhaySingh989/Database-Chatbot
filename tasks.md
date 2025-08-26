# DataSuperAgent: Production-Ready Development Plan

## How to Read This Document

This is a living document that outlines the requirements and development tasks for transforming the DataSuperAgent from an MVP into a production-ready application. It is designed to provide clear, actionable instructions for any developer.

**Structure:**

1.  **Section 1: Requirements (User Stories):** This section describes the *why* behind the work. It is organized into **Epics**, which are high-level project goals. Each Epic contains specific **User Stories**, which describe a feature from a user's perspective. Each story is assigned a **Priority** (High, Medium, Low).

2.  **Section 2: Sequential Task List:** This section describes the *how*. It provides a checklist of concrete development tasks. The tasks are grouped by Epic and should be completed in the order they are listed to ensure a stable development process.

**Development Philosophy:**
Work should be completed in order of priority, starting with **Epic 1: Foundational Robustness & Testing**. This epic addresses the core stability and testing foundation that is essential before significant new features are added.

---

## Section 1: Requirements (User Stories)

### Epic 1: Foundational Robustness & Testing

**Goal:** To establish a comprehensive testing framework and ensure the existing application is stable, reliable, and ready for future development. This is the highest priority.

---

**User_Story: 1.1**
As a developer, I want a comprehensive test suite for the application, so that I can verify existing functionality, catch bugs, and prevent regressions.
*Status: To Do*
*Priority: High*

#### Acceptance Criteria
1.  A testing framework (e.g., `pytest`) SHALL be integrated into the project.
2.  Unit tests SHALL be created for core business logic in `database_handler.py`, `agent_handler.py`, and `data_handler.py`.
3.  The tests SHALL cover critical functions, edge cases, and error handling.
4.  A CI/CD pipeline (e.g., GitHub Actions) SHALL be configured to run tests automatically on each push.

---

### Epic 2: Enhanced Data Connectivity

**Goal:** To expand the application's capabilities to connect directly to live databases, moving beyond simple file uploads.

---

**User_Story: 2.1**
As a data analyst, I want to connect the application directly to a live PostgreSQL database, so that I can analyze real-time data without needing to manually export and upload files.
*Status: To Do*
*Priority: High*

#### Acceptance Criteria
1.  The UI SHALL include a form to securely input database connection details (host, port, user, password, database name).
2.  The application SHALL use these credentials to establish a live connection to a PostgreSQL database.
3.  The application SHALL be able to fetch schema information and execute queries on the connected database.
4.  Credentials SHALL be handled securely and not exposed in the frontend.

---

**User_Story: 2.2**
As a data scientist, I want to connect to cloud data warehouses like Google BigQuery and Snowflake, so that I can leverage the application's analytical capabilities on large-scale enterprise data.
*Status: To Do*
*Priority: Medium*

#### Acceptance Criteria
1.  The UI SHALL be extended to support credential input for BigQuery (Service Account JSON) and Snowflake.
2.  The `database_handler.py` module SHALL be refactored to include specific connectors for BigQuery and Snowflake.
3.  The application SHALL be able to perform schema discovery and query execution on these cloud data warehouses.

---

### Epic 3: Advanced Analytics & AI Augmentation

**Goal:** To make the AI agent more proactive and insightful, providing users with automated analysis and deeper understanding of their data.

---

**User_Story: 3.1**
As a user, I want the application to automatically generate a data profile when I upload or connect to a data source, so that I can quickly understand its structure, quality, and key statistics.
*Status: To Do*
*Priority: High*

#### Acceptance Criteria
1.  Upon successful data connection, a "Data Profile" tab or section SHALL be populated.
2.  The profile SHALL include, for each column: data type, missing value count/percentage, and summary statistics (mean, median, std dev for numeric; unique counts for categorical).
3.  The profile SHALL include a correlation matrix visualization for numeric columns.

---

**User_Story: 3.2**
As a business user, I want the AI agent to proactively suggest interesting questions or highlight potential anomalies in my data, so that I can discover insights I might have missed.
*Status: To Do*
*Priority: Medium*

#### Acceptance Criteria
1.  After analyzing the schema, the agent SHALL generate a list of 3-5 suggested analytical questions (e.g., "What is the trend of sales over time?", "How are customers distributed by region?").
2.  The user SHALL be able to click on a suggestion to execute it as a query.
3.  The agent MAY identify columns with high cardinality or potential outliers and bring them to the user's attention.

---

### Epic 4: Interactive Dashboarding & Visualization

**Goal:** To empower users to create, customize, and share dynamic dashboards based on their data.

---

**User_Story: 4.1**
As a user, I want to generate a dashboard with key charts and KPIs with a single click, so that I can get a quick, high-level overview of my data.
*Status: To Do*
*Priority: High*

#### Acceptance Criteria
1.  A "Generate Dashboard" button SHALL be available after a data source is connected.
2.  Clicking the button SHALL trigger the AI agent to identify key metrics and dimensions in the data.
3.  The agent SHALL automatically generate a dashboard containing a mix of KPIs (e.g., total records, key averages), bar charts, line charts, and tables.

---

**User_Story: 4.2**
As a user, I want to customize the auto-generated dashboard, so that I can tailor it to my specific analytical needs.
*Status: To Do*
*Priority: Medium*

#### Acceptance Criteria
1.  The dashboard SHALL be modular, with each chart or KPI as a separate component.
2.  The user SHALL be able to remove, resize, and drag-and-drop components within the dashboard grid.
3.  The user SHALL be able to edit a chart, changing the chart type, metrics, or dimensions.

---

## Section 2: Sequential Task List

### Epic 1: Foundational Robustness & Testing
**Priority: High**
- [ ] **1. Set Up Testing Framework**
  - [ ] Add `pytest` to `requirements.txt`.
  - [ ] Create a `tests/` directory with an initial structure (e.g., `tests/test_handlers.py`).
  - [ ] Configure `pytest` by creating a `pytest.ini` or `pyproject.toml` file.
- [ ] **2. Write Unit Tests for Core Logic**
  - [ ] Write tests for `data_handler.py` to verify file parsing (CSV, Excel).
  - [ ] Write tests for `database_handler.py` using a mock or temporary SQLite DB to test schema fetching and query execution.
  - [ ] Write tests for `agent_handler.py` to ensure the AI agent is initialized correctly.
  - _User_Story: 1.1_

### Epic 2: Enhanced Data Connectivity
**Priority: High**
- [ ] **3. Refactor Database Handling**
  - [ ] Create a base `Database` abstract class in `database_handler.py` that defines a common interface (connect, get_schema, execute_query).
  - [ ] Refactor the existing SQLite logic into a `SQLiteHandler` class that inherits from the base class.
- [ ] **4. Implement PostgreSQL Connector**
  - [ ] Add `psycopg2-binary` to `requirements.txt`.
  - [ ] Create a `PostgreSQLHandler` class that implements the `Database` interface.
  - [ ] In `ui.py`, add a new section to the sidebar for selecting the data source type ("File Upload" vs. "Database Connection").
  - [ ] Create a Streamlit form to securely collect and use PostgreSQL connection details.
  - _User_Story: 2.1_
- [ ] **5. Implement Cloud Warehouse Connectors**
  - [ ] Add `google-cloud-bigquery` and `snowflake-connector-python` to `requirements.txt`.
  - [ ] Create `BigQueryHandler` and `SnowflakeHandler` classes.
  - [ ] Extend the UI to include forms for BigQuery and Snowflake authentication.
  - _User_Story: 2.2_

### Epic 3: Advanced Analytics & AI Augmentation
**Priority: High**
- [ ] **6. Implement Automated Data Profiling**
  - [ ] Integrate a profiling library like `pandas-profiling` or build a custom profiling function in `data_manager.py`.
  - [ ] Add a new tab to the main UI area that appears after data is loaded.
  - [ ] Call the profiling function and display the results (summary stats, missing values, etc.) in the new tab.
  - _User_Story: 3.1_
- [ ] **7. Implement Proactive AI Suggestions**
  - [ ] In `agent_handler.py`, create a new prompt for the Gemini model that asks it to generate 3-5 insightful questions based on a table schema.
  - [ ] After the schema is loaded, call the agent with this new prompt.
  - [ ] Display the suggested questions in the UI as clickable buttons that populate the query input box.
  - _User_Story: 3.2_

### Epic 4: Interactive Dashboarding & Visualization
**Priority: High**
- [ ] **8. Implement One-Click Dashboard Generation**
  - [ ] Create a new prompt for the agent that asks it to identify the most important KPIs and suggest visualizations (e.g., "Bar chart of sales by region," "Time series of user signups").
  - [ ] Add a "Generate Dashboard" button to the UI.
  - [ ] When clicked, send the schema to the agent with the new prompt. The agent should return a structured list of chart definitions (e.g., `{'type': 'bar', 'x': 'region', 'y': 'sales'}`).
  - [ ] Parse the agent's response and use Streamlit's native charting functions (`st.bar_chart`, `st.line_chart`, `st.metric`) to render the dashboard.
  - _User_Story: 4.1_
- [ ] **9. Add Dashboard Customization**
  - [ ] Investigate a library like `streamlit-dashboard` to create a draggable and resizable grid layout.
  - [ ] Refactor the dashboard generation to render each chart as a separate component within the grid.
  - [ ] Add "edit" and "delete" icons to each dashboard component.
  - _User_Story: 4.2_
