---
title: Database Notes
---

# Database Fundamentals

A **database** is an organized collection of data that can be stored, queried and updated efficiently. Most applications, from small mobile apps to large banking systems, rely on a database to keep their information safe between sessions. This note covers the three ideas every beginner should understand: indexes, transactions and normalization.

## Indexes

An index is a separate data structure that lets the database find rows without scanning the whole table. Think of the index at the back of a textbook: instead of reading every page to find the word "recursion", you look it up and jump straight to the right page.

Most relational databases use a **B-tree** index by default. A B-tree keeps keys in sorted order, so lookups, range queries and sorting are all fast. The cost is that every insert or update must also update the index, which makes writes slightly slower and uses extra disk space.

```sql
CREATE INDEX idx_students_email ON students (email);
```

A good rule of thumb is to index columns that appear often in `WHERE`, `JOIN` and `ORDER BY` clauses, and to avoid indexing columns that change constantly or have very few distinct values.

## Transactions

A transaction groups several operations into one unit of work that either fully succeeds or fully fails. The classic example is a bank transfer: subtracting money from one account and adding it to another must happen together, otherwise money would disappear or be created.

Transactions are described by the **ACID** properties. Atomicity means all or nothing. Consistency means the database moves from one valid state to another. Isolation means concurrent transactions do not see each other's half-finished work. Durability means that once a transaction is committed, it survives a crash or power loss.

Databases implement isolation with locks or with multi-version concurrency control (MVCC), where each transaction reads a consistent snapshot of the data. Stronger isolation levels prevent more anomalies but reduce how many transactions can run at the same time.

## Normalization

Normalization is the process of organizing tables to reduce duplicated data. Instead of storing a student's department name in every row of an enrollment table, you store a department id and keep the name in a separate departments table. When the name changes, you update it in exactly one place.

The first three normal forms are the most common. First normal form requires that each column holds a single value. Second normal form removes partial dependencies on a composite key. Third normal form removes columns that depend on other non-key columns.

Over-normalizing has a cost, because queries need more joins. Many analytics systems deliberately **denormalize** data to make reads faster, accepting some duplication in exchange for speed.
