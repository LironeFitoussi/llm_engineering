import sqlite3

DB = "prices.db"
with sqlite3.connect(DB) as conn:
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS prices (city TEXT PRIMARY KEY, price REAL)")
    conn.commit()
    
def get_ticket_price(city):
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT price FROM prices WHERE city=?", (city.lower(),))
        result = cursor.fetchone()
        # print(result[0])
        return f"The price of a ticket to {city} is {result[0]}" if result else "No price found for {city}."

def set_ticket_price(city, price):
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO prices (city, price) VALUES (?, ?) ON CONFLICT(city) DO UPDATE SET price = ?', (city.lower(), price, price))
        conn.commit()
        
ticket_prices = {
    "london":799,
    "paris": 899,
    "tokyo": 1420,
    "sydney": 2999,
    "tel aviv": 1299,
}

for city, price in ticket_prices.items():
    set_ticket_price(city, price)