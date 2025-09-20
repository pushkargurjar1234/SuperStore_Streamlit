import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

plt.style.use("dark_background")   
# plt.style.use("seaborn-v0_8-dark")



st.set_page_config(layout="wide",page_title="SuperStore Analysis")
df=pd.read_csv("Global_Superstore2.csv",encoding="latin1")
df.drop("Postal Code",axis=1,inplace=True)
df["Profit_status"]=df["Profit"].apply(lambda x: "Profitable" if x>0 else "Unprofitable" )
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, errors='coerce')
df["Year"]=df["Order Date"].dt.year
df["month"]=df["Order Date"].dt.month
df["YoY Growth %"] = df.groupby("Category")["Sales"].pct_change() * 100
bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
labels=["0-10%","10-20%","20-30%","30-40%","40-50%","50-60%","60-70%","70-80%","80-90%"]

df["Discount Range"]= pd.cut(df["Discount"],bins=bins,labels=labels,include_lowest=True)
df['Quarter'] = df['Order Date'].dt.quarter
df["Half"] = np.where(df["Order Date"].dt.month <= 6, "1st Half", "2nd Half")





                                # OverAll Analysis   --------------------------------------------------------------
def Overall_Analysis():
    st.title("overall")

    st.divider()

    col1,col2,col3=st.columns(3)
    
    with col1:
        Revenue=int(df["Sales"].sum())
        Revenue_million = round(Revenue / 1_000_000, 2)
        st.metric(label="Total Revenue", value="$"+str(Revenue_million)+"M")

    with col2:
        T_Profit=int(df["Profit"].sum())
        T_Profit_million=round(T_Profit/1_000_000, 2)
        st.metric(label="Total Profit", value="$"+str(T_Profit_million)+"M")
        
    with col3:
        T_Orders=int(df["Order ID"].count())
        st.metric(label="Total Order", value=T_Orders)

    st.write("")
    st.text(" ")

    col4,nouse1,col5=st.columns(3)

    with col4:
        Margin=round(((T_Profit/Revenue)*100),2)
        st.metric(label="Overall Profit margin for the Company", value=str(Margin)+"%")
        st.write(" ")

    with col5:
        U_Coustomer=df["Customer ID"].nunique()
        st.metric(label="Total Customer", value=str(U_Coustomer))

    st.text(" ")
    st.text(" ")
    col6,nouse2,col7=st.columns(3)

    with col6:
        U_Product=df["Product ID"].nunique()
        st.metric(label="Total Product", value=str(U_Product)+"%")
    
    with col7:
        Aov=round((Revenue/T_Orders),2)
        st.metric(label="Average order value (AOV)", value=str(Aov))
    
    st.divider()
    st.write()
    st.subheader("percentage of orders are profitable vs unprofitable")
    col8,nouse3=st.columns(2)
    with col8:
        profit_count=df["Profit_status"].value_counts()
        plt.pie(profit_count.values,autopct="%.1f%%",
        labels=profit_count.index,
        colors=["green","red"] )
        st.pyplot(plt.gcf())   

    st.divider()
    st.write()
    
    st.subheader("Most Successful Year in Terms of Sales and Profit")
    col9,nouse4=st.columns([3,1])
    with col9:
        S_year=df.groupby("Year")[["Profit","Sales"]].sum().reset_index()
        yearly_suc=S_year.melt(id_vars="Year",value_vars=['Profit','Sales'],var_name="Metric",value_name="Amount") 
        sns.set_style("white")
        plt.figure(facecolor='none')
        sns.barplot(data=yearly_suc,x="Year",y="Amount", hue="Metric", color="red")

        plt.xticks( color="white")
        plt.yticks( color="white")

    # plt.title("Successful Years in Terms of Sales and Profit", fontsize=14, color="white")
        plt.xlabel("Year", fontsize=12, color="white")
        plt.ylabel("Amount", fontsize=12, color="white")
        st.pyplot(plt)


                    #  Sales_Profit_Performance ------------------------------------------------------------------




def Sales_Profit_Performance():
    st.header("Sales Profit Performance")
    st.divider()

    
    col10,nouse5,col11=st.columns(3)
    with col10:
        Revenue=int(df["Sales"].sum())
        Revenue_million = round(Revenue / 1_000_000, 2)
        st.metric(label="Total Revenue", value="$"+str(Revenue_million)+"M")

    with col11:
        T_Profit=int(df["Profit"].sum())
        T_Profit_million=round(T_Profit/1_000_000, 2)
        st.metric(label="Total Profit", value="$"+str(T_Profit_million)+"M")

    st.divider()
    st.write()

    st.subheader('Identifying the Best and Worst Performing Months in Sales')
    col12,nouse5=st.columns([3,1])

    with col12:
        HM_Sale=df.groupby(["month","Year"])["Sales"].sum().unstack(fill_value=0)
        S_months=HM_Sale.index
        S_year=HM_Sale.columns
        colors = ['blue','orange','green','red']
        bottom = [0]*len(S_months)

        plt.figure(figsize=(12,6))
        for i ,Year in enumerate(S_year):
            plt.bar(S_months,HM_Sale[Year],bottom=bottom,color=colors[i],label=Year)
            bottom+=HM_Sale[Year]

        plt.xlabel("Month")
        plt.ylabel("Sales")
        plt.title("Stacked Sales by Year for Each Month")
        plt.xticks(S_months,color="black")
        plt.legend(title="Year")

        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.5)
        st.pyplot(plt)


    st.divider()
    st.write()


    st.subheader("Sales vs Profit Trend Over Time")
    col13,nouse=st.columns([3,1])
    with col13:
        S_trend=df.groupby("Year")[["Sales","Profit"]].sum().reset_index()
        S_trendMelated=df.melt(id_vars="Year",value_vars=["Sales","Profit"],value_name="Amount",var_name="Metric")
        figure,ax1=plt.subplots(figsize=(8,5))

        sns.lineplot(data=S_trend,x="Year",y="Sales",marker="o", ax=ax1,color="blue",label="Sales")
        plt.title("Sales vs Profit Over Time")

        ax1.set_ylabel("Sales", color="blue")
        ax1.tick_params(axis='y', labelcolor="blue")


        ax2 = ax1.twinx()
        sns.lineplot(data=S_trend,x="Year",y="Profit",marker="o", ax=ax1,color="red",label="Profit")
        ax2.set_ylabel("Profit", color="red")
        ax2.tick_params(axis='y', labelcolor="red")
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.5)
        st.pyplot(plt)

    st.divider()
    st.write()

    st.subheader("Products Driving the Highest Sales")
    col14,nouse=st.columns([3,1])

    with col14:
        TopProduct=df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(10).reset_index()
        products = TopProduct["Product Name"]
        sales = TopProduct["Sales"]

        plt.figure(figsize=(10,6))
        plt.barh(products, sales, color="red")
        plt.xlabel("Total Sales")
        plt.ylabel("Product Name")
        plt.title("Top 10 Products by Sales")
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.5)
        st.pyplot(plt)

    st.divider()
    st.write()


    st.subheader("Products Generating High Sales but Low Profit")
    col15,nouse=st.columns([3,1])
    with col15:
        Hslp=df.groupby("Product Name")[["Sales","Profit"]].sum().reset_index()
        S_thresehold=Hslp["Sales"].mean()
        P_thresehold=Hslp["Profit"].mean()
        high_slp=Hslp[(Hslp["Sales"]>S_thresehold)
              & (Hslp["Profit"]<P_thresehold) ]
        plt.figure(figsize=(10,5))
        sns.scatterplot(data=high_slp ,x="Profit", y="Sales",color="brown")
        plt.xlabel("Total Profit")
        plt.ylabel("Total_Sales")
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.5)
        st.pyplot(plt)
    

#                                   Regional_State_Analysis  -------------------------------------------------------------



def Regional_State_Analysis():
    st.header("Regional State Analysis")
    
    st.divider()


    st.subheader("Region Contributing the Most to Total Sales")
    col1,nouse=st.columns([3,1])
    with col1:
        T_Region=df.groupby("Region")["Sales"].sum().sort_values(ascending=False).head(3).reset_index()
        plt.figure(figsize=(8,5))
        sns.barplot(data=T_Region ,x="Region",y="Sales",color="green",palette="viridis")
        plt.title("Most Contributed Regions ")
        plt.xlabel("Total Sales")
        plt.ylabel("Regions")
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.5)
        st.pyplot(plt)

    st.divider()
    st.write()


    st.subheader("Region Achieving Maximum Profit Percentage")
    col2,nouse=st.columns([1.5,1])
    with col2:
        H_Profit=df.groupby("Region")[["Profit","Sales"]].sum().reset_index()
        H_Profit["Percent"]=(H_Profit["Profit"]/H_Profit["Sales"])*100
        H_Profit.sort_values(by="Percent",ascending=False)
        plt.figure(figsize=(10,6))
        plt.pie(H_Profit["Percent"],
        labels=H_Profit["Region"],
        autopct="%.1f%%")
        plt.title("Profit Percentage by Region")
        st.pyplot(plt)

        st.divider()
        st.write()

    st.subheader("States Driving the Highest Sales – Top 5")
    col3,nouse=st.columns([3,1])
    with col3:
        T_State=df.groupby("State")["Sales"].sum().sort_values(ascending=False).head(5).reset_index()
        plt.figure(figsize=(10,6))
        sns.barplot(data=T_State ,y="State",x="Sales",palette="magma")
        plt.title("Top 5 States by Sales", fontsize=14)
        plt.xlabel("State")
        plt.ylabel("Total Sales")
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.5)
        st.pyplot(plt)

    
    st.divider()
    st.write()

    st.subheader('Regions/States with High Sales but Low Profit')
    col4,nouse=st.columns([3,1])
    with col4:
        H_Region=df.groupby("Region")[["Sales","Profit"]].sum().reset_index()
        H_Sale=H_Region['Sales'].mean()
        L_Sale=H_Region['Profit'].mean()
        HL_Region=H_Region[(H_Region["Sales"]>H_Sale) & H_Region["Profit"]<L_Sale]
        plt.figure(figsize=(10,6))
        sns.scatterplot(data=HL_Region, x="Profit", y="Sales", s=100, hue="Region", palette="tab10", edgecolor="black")
        plt.title("Sales vs Profit by State", fontsize=16)
        plt.xlabel("Total Profit", fontsize=14)
        plt.ylabel("Total Sales in 10^6", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim(0, HL_Region["Profit"].max()*1.1)
        plt.legend(title="Region", fontsize=10, title_fontsize=12)
        plt.tight_layout()
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.5)
        st.pyplot(plt)


    st.divider()
    st.write()

    st.subheader('How Average Order Value Differs by Region')
    col5,nouse=st.columns([3,1])
    with col5:
        Region_Aov=df.groupby("Region").agg(
        T_Sale=("Sales","sum"),
        T_Order=("Order ID","nunique")).reset_index()
        Region_Aov["Aov"]=Region_Aov["T_Sale"]/Region_Aov["T_Order"]
        plt.figure(figsize=(10,6))
        sns.barplot(x="Region", y="Aov", data=Region_Aov, palette="viridis")

        plt.xticks(rotation=45)
        plt.title("Average Order Value Across Regions", fontsize=14)
        plt.ylabel("Average Order Value")
        plt.xlabel("Region")
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.5)
        st.pyplot(plt)

#                                       Category/Sub-Category Insights  -----------------------------------------              


def Category_Sub_Category_Insights():
    st.header("Category Sub-Category Insights")

    st.divider()

    st.subheader('Product Category Contribution to Revenue')
    col1,nouse=st.columns([3,1])
    with col1:
        Most_D = df.groupby("Category")["Sales"].sum()
        D_Category = int(Most_D.iloc[0])   
        H_Margin=df.groupby("Sub-Category")[["Sales","Profit"]].sum().reset_index()


        H_Margin["percent"] = (H_Margin["Profit"] / H_Margin["Sales"]) * 100

        plt.figure(figsize=(12, 6))
        sns.barplot(
        data=H_Margin,
        y="Sub-Category",
        x="percent",
        palette="plasma"
        )

        plt.title("Profit Margin by Sub-Category", fontsize=14)
        plt.xlabel("Profit Margin (%)")
        plt.ylabel("Sub-Category")

        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.5)

        st.pyplot(plt)
        


    st.divider()
    st.write()

    st.subheader("Sub-Categories with High Discount but Low Profit")
    col2,nouse=st.columns([3,1])
    with col2:
        BadDiscount = df.groupby("Sub-Category")[["Discount","Profit"]].sum().reset_index()
        AvgDiscount = BadDiscount["Discount"].mean()
        AvgProfit = BadDiscount["Profit"].mean()

        BadProfit = BadDiscount[
        (BadDiscount["Discount"] > AvgDiscount) &
        (BadDiscount["Profit"] < AvgProfit)
        ]

        plt.figure(figsize=(10, 6))
        sns.barplot(
        data=BadProfit,
        x="Sub-Category",
        y="Profit",
        palette="coolwarm"
        )

        plt.title("High Discount but Low Profit Sub-Categories", fontsize=14)
        plt.xlabel("Sub-Category")
        plt.ylabel("Profit")

        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.5)

        st.pyplot(plt)


    st.divider()
    st.write()

    st.subheader("Year-on-Year Sales Growth by Category")
    col3,nouse=st.columns([3,1])
    with col3:
        df["YoY Growth %"] = df.groupby("Category")["Sales"].pct_change() * 100


        plt.figure(figsize=(10, 6))
        sns.lineplot(
        data=df,
        x="Year",
        y="YoY Growth %",
        hue="Category",
        marker="o"
        )
        plt.axhline(0, color="gray", linestyle="--")

        plt.title("Year-on-Year Sales Growth by Category", fontsize=14)
        plt.ylabel("Growth (%)")
        plt.xlabel("Year")

        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(0.5)

        st.pyplot(plt)


                                                # Customer Insights ---------------------------------------------


def Customer_Insights():
    st.header("Customer Insights")

    st.divider()
    st.write()

    st.subheader("Top 10 Customers by Total Sales")
    col1,nouse=st.columns([3,1])
    with col1:
        top_customers = df.groupby(["Customer ID", "Customer Name"])["Sales"].sum().sort_values(ascending=False).head(10)
        top_customers_df = top_customers.reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hlines(
        y=top_customers_df["Customer Name"], 
        xmin=0, 
        xmax=top_customers_df["Sales"], 
        color="turquoise"
        )
        ax.plot(
    top_customers_df["Sales"], 
    top_customers_df["Customer Name"], 
    "o", 
    color="navy"
    )

        ax.set_title("Top 10 Customers by Total Sales", fontsize=14, fontweight="bold")
        ax.set_xlabel("Total Sales")
        ax.set_ylabel("Customer Name")
        ax.grid(axis="x", linestyle="--", alpha=0.6)

        st.pyplot(fig)


    st.divider()
    st.write()

    st.subheader("Customer Segment with Highest Spending")

    col2,nouse=st.columns(2)
    with col2:

        Customer_SM = df.groupby("Segment")["Sales"].sum().reset_index()
        segments = Customer_SM["Segment"]
        sales = Customer_SM["Sales"]
        colors = ["indigo", "sienna", "darkolivegreen"]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(
    sales, 
    labels=segments, 
    autopct="%.1f%%", 
    startangle=90, 
    colors=colors
)
        ax.set_title("Total Sales by Segment", fontsize=14, fontweight="bold")

        st.pyplot(fig)


    st.divider()
    st.write()
    st.subheader("Top Customers Based on Number of Orders Placed")
    col3,nouse=st.columns([3,1])
    with col3:
        Max_O = df.groupby(["Customer ID", "Customer Name"])["Order ID"].count().sort_values(ascending=False).head(10)
        Max_O_df = Max_O.reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(Max_O_df["Customer Name"], Max_O_df["Order ID"], color="steelblue")

        ax.set_title("Top 10 Customers by Number of Orders", fontsize=14, fontweight="bold")
        ax.set_xlabel("Customer Name")
        ax.set_ylabel("Number of Orders")
        ax.set_xticklabels(Max_O_df["Customer Name"], rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        plt.tight_layout()
        st.pyplot(fig)



    st.divider()
    st.write()

    st.subheader("Average Order Value (AOV) by Customer Segment")
    col4,nouse=st.columns(2)
    with col4:
        Customer_AOV = df.groupby("Segment")[["Sales","Order ID"]].agg({"Sales":"sum", "Order ID":"count"}).reset_index()
        Customer_AOV["AOV"] = Customer_AOV["Sales"] / Customer_AOV["Order ID"]

        segments = Customer_AOV["Segment"]
        aov = Customer_AOV["AOV"]
        colors = ["teal", "darkorange", "orchid"]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(
    aov, 
    labels=segments, 
    autopct="%.1f%%", 
    startangle=90, 
    colors=colors, 
    wedgeprops=dict(width=0.4),
    pctdistance=0.75
        )
        st.pyplot(fig)


#                                           Discount Order Analysis  ---------------------------------




def Discount_Order_Analysis():
    st.header("Discount Order Analysis")

    

    st.divider()
    st.write()

    st.subheader("Impact of Discount on Sales & Profit")
    col1,nouse=st.columns([3,1])
    with col1:

        Isrp = df.groupby("Discount")[["Sales", "Profit"]].sum().reset_index()

        fig, ax1 = plt.subplots(figsize=(10, 6))


        ax1.set_xlabel("Discount")
        ax1.set_ylabel("Total Sales", color="blue")
        ax1.plot(Isrp["Discount"], Isrp["Sales"], label="Sales", marker="*", color="blue")


        ax2 = ax1.twinx()
        ax2.set_ylabel("Total Profit", color="red")
        ax2.plot(Isrp["Discount"], Isrp["Profit"], label="Profit", marker="o", color="red")


        plt.title("Impact of Discount on Sales & Profit", fontsize=14)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        st.pyplot(fig)




    st.divider()
    st.write()

    st.subheader("Sales & Profit by Discount Range")
    col3,nouse=st.columns([3,1])
    with col3:

        Discount_bar = df.groupby("Discount Range")[["Sales", "Profit"]].sum()

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(Discount_bar))
        width = 0.35

        ax.bar(x - width/2, Discount_bar["Sales"], width, label="Sales", color="lightblue")
        ax.bar(x + width/2, Discount_bar["Profit"], width, label="Profit", color="pink")

        ax.set_xticks(x)
        ax.set_xticklabels(Discount_bar.index)
        ax.set_xlabel("Discount Range")
        ax.set_ylabel("Amount")
        ax.set_title("Sales & Profit by Discount Range")
        ax.legend()

        st.pyplot(fig)


    st.divider()
    st.write()

    st.subheader("Category-wise Sales vs Profit by Discount Range")
    col4,nouse=st.columns([3,1])
    with col4:
        Dpc = df.groupby(["Category", "Discount Range"])[["Sales", "Profit"]].sum().reset_index()
        categories = Dpc["Category"].unique()
        x = range(len(labels))

        for cat in categories:
            cat_data = Dpc[Dpc["Category"] == cat]
            sales = cat_data.set_index("Discount Range").reindex(labels)["Sales"]
            profit = cat_data.set_index("Discount Range").reindex(labels)["Profit"]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(x, sales, width=0.4, label="Sales", color="blue", align="center")
            ax.bar([p + 0.4 for p in x], profit, width=0.4, label="Profit", color="red", align="center")

            ax.set_xticks([p + 0.2 for p in x])
            ax.set_xticklabels(labels, rotation=45, fontsize=11, fontweight="bold")
            ax.set_title(f"{cat}: Sales vs Profit by Discount Range", fontsize=14)
            ax.set_xlabel("Discount Range")
            ax.set_ylabel("Amount")
            ax.legend()
            plt.tight_layout()

            st.pyplot(fig)



    st.divider()
    st.write()

    st.subheader("Top 7 Products with Highest Losses")
    col5,nouse=st.columns([3,1])
    with col5:
            loss_orders = df[df["Profit"] < 0][["Profit", "Product Name"]].sort_values(by="Profit").head(7)

            
            plt.figure(figsize=(10, 6))
            colors = plt.cm.viridis(np.linspace(0, 1, len(loss_orders)))

            plt.bar(loss_orders["Product Name"], abs(loss_orders["Profit"]), color=colors)
            plt.xlabel("Product")
            plt.ylabel("Profit (Loss)")
            plt.xticks(rotation=45, ha="right")
            plt.title("Top 7 Products with Highest Losses", fontsize=14)

            st.pyplot(plt)


#                                                      Time-based Analysis  ----------------------------------------

def Time_based_Analysis():
    st.header("Time-based Analysis")



    st.divider()
    st.write()

    st.subheader("Sales vs Profit Trend Over Years")

    col1,nouse=st.columns([3,1])
    with col1:

        Year_trend = df.groupby("Year")[["Sales", "Profit"]].sum().reset_index()

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(Year_trend["Year"], Year_trend["Sales"], color="red", marker='o', label="Sales")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Sales", color="red")
        ax1.tick_params(axis='y', labelcolor="red")

        ax2 = ax1.twinx()
        ax2.plot(Year_trend["Year"], Year_trend["Profit"], color="blue", marker='s', label="Profit")
        ax2.set_ylabel("Profit", color="blue")
        ax2.tick_params(axis='y', labelcolor="blue")

        plt.title("Sales vs Profit Trend Over Years", fontsize=14)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

        st.pyplot(fig)




    st.divider()
    st.write()

    col2,nouse=st.columns([3,1])
    with col2:

        st.subheader("Best Quarter by Sales")

        Best_Quarter = df.groupby("Quarter")["Sales"].sum().reset_index()

        plt.figure(figsize=(8, 5))
        plt.bar(Best_Quarter["Quarter"], Best_Quarter["Sales"], color="skyblue")

        plt.xlabel("Quarter")
        plt.ylabel("Total Sales")
        plt.title("Sales by Quarter", fontsize=14)

        st.pyplot(plt)



    st.divider()
    st.write()
    col3,nouse=st.columns([3,1])
    with col3:


        st.subheader("Monthly Sales Distribution")

        higest_month = df.groupby("month")["Sales"].sum().reset_index()
        months = higest_month["month"].tolist()
        sales = higest_month["Sales"].tolist()
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

        plt.figure(figsize=(8, 8))
        plt.pie(
        sales,
        labels=month_names,
        autopct="%1.1f%%",
        startangle=90,
        colors=plt.cm.tab20.colors
        )
        # plt.title("Monthly Sales Distribution", fontsize=14)
        plt.axis("equal")

        st.pyplot(plt)





    st.divider()
    st.write()

    st.subheader("Monthly Sales by Category (Seasonality View)")
    col4,nouse=st.columns([3,1])
    with col4:
        monthly_category_sales = df.groupby(["Category", "month"])["Sales"].sum().reset_index()
        plt.style.use("dark_background")
        plt.figure(figsize=(12, 6))
        for category in monthly_category_sales["Category"].unique():
            subset = monthly_category_sales[monthly_category_sales["Category"] == category]
            plt.plot(subset["month"], subset["Sales"], marker="o", label=category)

        plt.title("Monthly Sales by Category (Seasonality View)", fontsize=14)
        plt.xlabel("Month")
        plt.ylabel("Sales")
        plt.legend(title="Category")
        plt.grid(True, linestyle="--", alpha=0.6)

        st.pyplot(plt)


    

    st.divider()
    st.write()
    st.subheader("Comparison of Profit Margins: First Half vs Second Half of the Year")

    col5, nouse = st.columns([3, 1])
    with col5:
        Half_Margin = df.groupby("Half")[["Sales", "Profit"]].sum()
        Half_Margin["margin"] = (Half_Margin["Profit"] / Half_Margin["Sales"]) * 100  

    # Pie chart
        fig, ax = plt.subplots()
        ax.pie(
        Half_Margin["margin"], 
        labels=Half_Margin.index, 
        autopct="%.2f%%", 
        colors=["gold", "skyblue"],
        startangle=90)

        st.pyplot(fig)





                                    # sidebar ---------------------------------------------------------

st.sidebar.header("SuperStore Analysis")

option=st.sidebar.selectbox("Select One",["Overall Analysis","Sales & Profit Performance","Regional / State Analysis","Category & Sub-Category Insights","Customer Insights","Discount & Order Analysis","Time-based Analysis"])

if option=="Overall Analysis":
    btn0=st.sidebar.button("Overall Analysis")
    if btn0:
        Overall_Analysis()
        
if option=="Sales & Profit Performance":
    btn1=st.sidebar.button("Sales & Profit Performance")
    if btn1:
        Sales_Profit_Performance()

if option=="Regional / State Analysis":
    btn0=st.sidebar.button("Regional / State Analysis")
    if btn0:
        Regional_State_Analysis()

if option=="Category & Sub-Category Insights":
    btn0=st.sidebar.button("Category & Sub-Category Insights")
    if btn0:
        Category_Sub_Category_Insights()

if option=="Customer Insights":
    btn0=st.sidebar.button("Customer Insights")
    if btn0:
        Customer_Insights()

if option=="Discount & Order Analysis":
    btn0=st.sidebar.button("Discount & Order Analysis")
    if btn0:
        Discount_Order_Analysis()

if option=="Time-based Analysis":
    btn0=st.sidebar.button("Time-based Analysis")
    if btn0:
        Time_based_Analysis()
