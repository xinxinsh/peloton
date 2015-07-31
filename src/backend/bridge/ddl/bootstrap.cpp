/*-------------------------------------------------------------------------
 *
 * bootstrap.cpp
 * file description
 *
 * Copyright(c) 2015, CMU
 *
 * /peloton/src/backend/bridge/bootstrap.cpp
 *
 *-------------------------------------------------------------------------
 */

#include <iostream>
#include <sys/types.h>
#include <unistd.h>
#include <cassert>

#include "backend/bridge/ddl/bridge.h"
#include "backend/bridge/ddl/bootstrap.h"
#include "backend/bridge/ddl/bootstrap_utils.h"
#include "backend/bridge/ddl/ddl_database.h"
#include "backend/bridge/ddl/ddl_table.h"
#include "backend/storage/database.h"
#include "backend/common/logger.h"

#include "access/heapam.h"
#include "access/htup_details.h"
#include "access/xact.h"
#include "catalog/pg_attribute.h"
#include "catalog/pg_constraint.h"
#include "catalog/pg_class.h"
#include "catalog/pg_database.h"
#include "catalog/pg_namespace.h"
#include "commands/dbcommands.h"
#include "common/fe_memutils.h"
#include "utils/ruleutils.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/snapmgr.h"
#include "utils/syscache.h"

namespace peloton {
namespace bridge {

std::vector<catalog::Column> Bootstrap::GetRelationColumns(
    Oid tuple_oid, Relation pg_attribute_rel) {
  HeapScanDesc pg_attribute_scan;
  HeapTuple pg_attribute_tuple;
  std::vector<catalog::Column> columns;

  // Scan the pg_attribute table for the relation oid we are interested in.
  pg_attribute_scan = heap_beginscan_catalog(pg_attribute_rel, 0, NULL);

  // Go over all attributes in "pg_attribute" looking for any entries
  // matching the given tuple oid.
  // For instance, this means the columns associated with a given relation oid.
  while (1) {
    Form_pg_attribute pg_attribute;

    // Get next <relation, attribute> tuple from pg_attribute table
    pg_attribute_tuple = heap_getnext(pg_attribute_scan, ForwardScanDirection);

    if (!HeapTupleIsValid(pg_attribute_tuple)) break;

    // Check the relation oid
    pg_attribute = (Form_pg_attribute)GETSTRUCT(pg_attribute_tuple);
    if (pg_attribute->attrelid == tuple_oid) {
      // Skip system columns in the attribute list
      if (strcmp(NameStr(pg_attribute->attname), "cmax") &&
          strcmp(NameStr(pg_attribute->attname), "cmin") &&
          strcmp(NameStr(pg_attribute->attname), "ctid") &&
          strcmp(NameStr(pg_attribute->attname), "xmax") &&
          strcmp(NameStr(pg_attribute->attname), "xmin") &&
          strcmp(NameStr(pg_attribute->attname), "tableoid")) {
        std::vector<catalog::Constraint> constraint_infos;

        PostgresValueType postgresValueType =
            (PostgresValueType)pg_attribute->atttypid;
        ValueType value_type =
            PostgresValueTypeToPelotonValueType(postgresValueType);
        int column_length = pg_attribute->attlen;
        bool is_inlined = true;
        if (pg_attribute->attlen == -1) {
          column_length = pg_attribute->atttypmod;
          is_inlined = false;
        }

        // NOT NULL constraint
        if (pg_attribute->attnotnull) {
          catalog::Constraint constraint(CONSTRAINT_TYPE_NOTNULL);
          constraint_infos.push_back(constraint);
        }

        // DEFAULT value constraint
        if (pg_attribute->atthasdef) {
          catalog::Constraint constraint(CONSTRAINT_TYPE_DEFAULT);
          constraint_infos.push_back(constraint);
        }

        catalog::Column column(value_type, column_length,
                               NameStr(pg_attribute->attname), is_inlined);
        columns.push_back(column);
      }
    }
  }

  heap_endscan(pg_attribute_scan);

  return columns;
}

void Bootstrap::CreateIndexInfos(oid_t tuple_oid, char *relation_name,
                                 const std::vector<catalog::Column> &columns,
                                 std::vector<IndexInfo> &index_infos) {
  Relation pg_index_rel;
  HeapScanDesc pg_index_scan;
  HeapTuple pg_index_tuple;

  pg_index_rel = heap_open(IndexRelationId, AccessShareLock);
  pg_index_scan = heap_beginscan_catalog(pg_index_rel, 0, NULL);

  // Go over the pg_index catalog table looking for indexes
  // that are associated with this table
  while (1) {
    Form_pg_index pg_index;

    pg_index_tuple = heap_getnext(pg_index_scan, ForwardScanDirection);
    if (!HeapTupleIsValid(pg_index_tuple)) break;

    pg_index = (Form_pg_index)GETSTRUCT(pg_index_tuple);

    // Search for the tuple in pg_index corresponding to our index
    if (pg_index->indexrelid == tuple_oid) {
      std::vector<std::string> key_column_names;

      for (auto column_info : columns) {
        key_column_names.push_back(column_info.column_name);
      }

      IndexType method_type = INDEX_TYPE_BTREE;
      IndexConstraintType type;

      if (pg_index->indisprimary) {
        type = INDEX_CONSTRAINT_TYPE_PRIMARY_KEY;
      } else if (pg_index->indisunique) {
        type = INDEX_CONSTRAINT_TYPE_UNIQUE;
      } else {
        type = INDEX_CONSTRAINT_TYPE_DEFAULT;
      }

      // Store all index information here
      // This is required because we can only create indexes at once
      // after all tables are created
      // The order of table and index entries in pg_class table can be arbitrary
      IndexInfo indexinfo(relation_name, pg_index->indexrelid,
                          get_rel_name(pg_index->indrelid), method_type, type,
                          pg_index->indisunique, key_column_names);

      index_infos.push_back(indexinfo);
      elog(LOG,"Create index %s on %s", indexinfo.GetIndexName().c_str(),
               indexinfo.GetTableName().c_str());
      break;
    }
  }

  heap_endscan(pg_index_scan);
  heap_close(pg_index_rel, AccessShareLock);
}

void Bootstrap::CreatePelotonStructure(
    char relation_kind, char *relation_name, Oid tuple_oid,
    const std::vector<catalog::Column> &columns,
    std::vector<IndexInfo> &index_infos) {
  bool status = false;

  switch (relation_kind) {
    // Create the Peloton table
    case 'r': {
      status = DDLTable::CreateTable(tuple_oid, relation_name, columns);

      if (status == true) {
        elog(LOG, "Create Table \"%s\" in Peloton", relation_name);
      } else {
        elog(ERROR, "Create Table \"%s\" in Peloton", relation_name);
      }

    } break;

    // Create the Peloton index
    case 'i': {
      CreateIndexInfos(tuple_oid, relation_name, columns, index_infos);
    } break;

    default:
      elog(ERROR, "Invalid pg_class entry type : %c", relation_kind);
      break;
  }
}

void Bootstrap::LinkForeignKeys(void) {
  Relation pg_constraint_rel;
  HeapScanDesc pg_constraint_scan;
  HeapTuple pg_constraint_tuple;

  oid_t database_oid = Bridge::GetCurrentDatabaseOid();
  assert(database_oid);

  pg_constraint_rel = heap_open(ConstraintRelationId, AccessShareLock);
  pg_constraint_scan = heap_beginscan_catalog(pg_constraint_rel, 0, NULL);

  // Go over the pg_constraint catalog table looking for foreign key constraints
  while (1) {
    Form_pg_constraint pg_constraint;

    pg_constraint_tuple =
        heap_getnext(pg_constraint_scan, ForwardScanDirection);
    if (!HeapTupleIsValid(pg_constraint_tuple)) break;

    pg_constraint = (Form_pg_constraint)GETSTRUCT(pg_constraint_tuple);

    // We only handle foreign key constraints here
    if (pg_constraint->contype != 'f') continue;

    // Extract oid
    Oid source_table_oid = pg_constraint->conrelid;
    assert(source_table_oid);
    Oid sink_table_oid = pg_constraint->confrelid;
    assert(sink_table_oid);

    auto &manager = catalog::Manager::GetInstance();
    auto source_table = manager.GetTableWithOid(database_oid, source_table_oid);
    assert(source_table);
    auto sink_table = manager.GetTableWithOid(database_oid, sink_table_oid);
    assert(sink_table);

    // TODO :: Find better way..
    bool isNull;
    Datum curr_datum =
        heap_getattr(pg_constraint_tuple, Anum_pg_constraint_conkey,
                     RelationGetDescr(pg_constraint_rel), &isNull);
    Datum ref_datum =
        heap_getattr(pg_constraint_tuple, Anum_pg_constraint_confkey,
                     RelationGetDescr(pg_constraint_rel), &isNull);

    ArrayType *curr_arr = DatumGetArrayTypeP(curr_datum);
    ArrayType *ref_arr = DatumGetArrayTypeP(ref_datum);
    int16 *curr_attnums = (int16 *)ARR_DATA_PTR(curr_arr);
    int16 *ref_attnums = (int16 *)ARR_DATA_PTR(ref_arr);
    int source_numkeys = ARR_DIMS(curr_arr)[0];
    int ref_numkeys = ARR_DIMS(ref_arr)[0];

    std::vector<std::string> pk_column_names;
    std::vector<std::string> fk_column_names;

    auto source_table_schema = source_table->GetSchema();
    auto sink_table_schema = sink_table->GetSchema();

    // Populate foreign key column names
    for (int source_key_itr = 0; source_key_itr < source_numkeys;
         source_key_itr++) {
      AttrNumber attnum = curr_attnums[source_key_itr];
      catalog::Column column = source_table_schema->GetColumn(attnum - 1);
      fk_column_names.push_back(column.GetName());
    }

    // Populate primary key column names
    for (int sink_key_itr = 0; sink_key_itr < ref_numkeys; sink_key_itr++) {
      AttrNumber attnum = ref_attnums[sink_key_itr];
      catalog::Column column = sink_table_schema->GetColumn(attnum - 1);
      pk_column_names.push_back(column.GetName());
    }

    std::string constraint_name = NameStr(pg_constraint->conname);

    auto foreign_key =
        new catalog::ForeignKey(sink_table_oid, pk_column_names,
                                fk_column_names, pg_constraint->confupdtype,
                                pg_constraint->confdeltype, constraint_name);

    source_table->AddForeignKey(foreign_key);
  }

  heap_endscan(pg_constraint_scan);
  heap_close(pg_constraint_rel, AccessShareLock);
}

/**
 * @brief create all databases using catalog table pg_database
 */
void Bootstrap::CreateDatabases() {
  Relation pg_database_rel;
  HeapScanDesc scan;
  HeapTuple tup;

  StartTransactionCommand();

  // Scan pg database table
  pg_database_rel = heap_open(DatabaseRelationId, AccessShareLock);
  scan = heap_beginscan_catalog(pg_database_rel, 0, NULL);

  while (HeapTupleIsValid(tup = heap_getnext(scan, ForwardScanDirection))) {
    Oid database_oid = HeapTupleHeaderGetOid(tup->t_data);
    DDLDatabase::CreateDatabase(database_oid);
  }

  heap_endscan(scan);
  heap_close(pg_database_rel, AccessShareLock);

  CommitTransactionCommand();
}

bool Bootstrap::BootstrapPeloton(void) {
  // Create the new storage database and add it to the manager
  CreateDatabases();

  // Relations for catalog tables
  Relation pg_class_rel;
  Relation pg_attribute_rel;

  HeapScanDesc pg_class_scan;
  HeapTuple pg_class_tuple;

  std::vector<IndexInfo> index_infos;
  bool status;

  elog(LOG, "Initializing Peloton");

  StartTransactionCommand();

  // Open the pg_class and pg_attribute catalog tables
  pg_class_rel = heap_open(RelationRelationId, AccessShareLock);
  pg_attribute_rel = heap_open(AttributeRelationId, AccessShareLock);

  pg_class_scan = heap_beginscan_catalog(pg_class_rel, 0, NULL);

  // Go over all tuples in "pg_class"
  // pg_class has info about tables and everything else that has columns or is
  // otherwise similar to a table.
  // This includes indexes, sequences, views, composite types, and some kinds of
  // special relation.
  // So, each tuple can correspond to a table, index, etc.
  while (1) {
    Form_pg_class pg_class;
    char *relation_name;
    char relation_kind;
    int attnum;

    // Get next tuple from pg_class
    pg_class_tuple = heap_getnext(pg_class_scan, ForwardScanDirection);

    if (!HeapTupleIsValid(pg_class_tuple)) break;

    pg_class = (Form_pg_class)GETSTRUCT(pg_class_tuple);
    relation_name = NameStr(pg_class->relname);
    relation_kind = pg_class->relkind;

    // Handle only user-defined structures, not pg-catalog structures
    if (pg_class->relnamespace != PG_PUBLIC_NAMESPACE) continue;

    // TODO: Currently, we only handle relations and indexes
    if (pg_class->relkind != 'r' && pg_class->relkind != 'i') {
      continue;
    }

    // We only support tables with atleast one attribute
    attnum = pg_class->relnatts;
    assert(attnum > 0);

    // Get the tuple oid
    // This can be a relation oid or index oid etc.
    Oid tuple_oid = HeapTupleHeaderGetOid(pg_class_tuple->t_data);

    auto columns = GetRelationColumns(tuple_oid, pg_attribute_rel);

    // Create peloton structure
    CreatePelotonStructure(relation_kind, relation_name, tuple_oid, columns,
                           index_infos);
  }

  // Create Indexes
  status = DDLIndex::CreateIndexes(index_infos);
  if (status == false) {
    elog(LOG, "Could not create an index in Peloton");
  }

  // Link foreign keys
  LinkForeignKeys();

  /*printf("Print all relation's schema information\n");
  auto& manager = catalog::Manager::GetInstance();
  storage::Database* db = manager.GetDatabaseWithOid(Bridge::GetCurrentDatabaseOid());
  std::cout << *db << std::endl;*/

  heap_endscan(pg_class_scan);
  heap_close(pg_attribute_rel, AccessShareLock);
  heap_close(pg_class_rel, AccessShareLock);

  CommitTransactionCommand();

  elog(LOG, "Finished initializing Peloton");

  return true;
}


//===--------------------------------------------------------------------===//
// NEW BOOTSTRAP FUNCTIONS
//===--------------------------------------------------------------------===//

/**
 * @brief Collecting information regarding tables, indexes, foreign key from
 * Postgres for bootstrap
 * @return raw structure
 */
raw_database_info* Bootstrap::GetRawDatabase(void){

  // Create and initialize raw database
  raw_database_info* raw_database = Bootstrap::InitRawDatabase();

  std::vector<raw_table_info*> raw_tables;
  std::vector<raw_index_info*> raw_indexes;
  std::vector<raw_foreignkey_info*> raw_foreignkeys;

  // Get Raw Tables and Indexes
  GetRawTableAndIndex(raw_tables, raw_indexes);

  // Get Raw Foreignkeys
  GetRawForeignKeys(raw_foreignkeys);

  BootstrapUtils::CopyRawTables(raw_database, raw_tables);
  BootstrapUtils::CopyRawIndexes(raw_database, raw_indexes);

  //BootstrapUtils::PrintRawDatabase(raw_database);

  return raw_database;
}

/**
 * @brief This function constructs all the user-defined tables and indices in all databases
 * @params raw_database raw data that contains information about tables,
 * indexes, foreign key to create in Peloton
 * @return true or false, depending on whether we could bootstrap.
 */
bool Bootstrap::NewBootstrapPeloton(raw_database_info* raw_database){
  // create the database with current database id
  elog(LOG, "Initializing database %s(%u) in Peloton", raw_database->database_name, raw_database->database_oid);
  DDLDatabase::CreateDatabase(raw_database->database_oid);

  BootstrapUtils::PrintRawDatabase(raw_database);

  // Create table
  CreateTables(raw_database->raw_tables, raw_database->table_count);

  // build indexes
  CreateIndexes(raw_database->raw_indexes, raw_database->index_count);

  auto& manager = catalog::Manager::GetInstance();
  storage::Database* db = manager.GetDatabaseWithOid(Bridge::GetCurrentDatabaseOid());
  std::cout << *db << std::endl;

  // link foreign keys
  elog(LOG, "Finished initializing Peloton");


  return true;
}

/**
 * @brief initialize raw database
 * @return raw database
 */
raw_database_info* Bootstrap::InitRawDatabase(){

  // Set databse oid and name
  raw_database_info* raw_database = (raw_database_info*)palloc(sizeof(raw_database_info));
  raw_database->database_oid = MyDatabaseId;
  raw_database->database_name = BootstrapUtils::CopyString(get_database_name( MyDatabaseId ));
  return raw_database;
}

/**
 * @brief construct raw tables and indexes
 * @param raw tables
 * @param raw indexes
 */
void Bootstrap::GetRawTableAndIndex(std::vector<raw_table_info*>& raw_tables,
                                    std::vector<raw_index_info*>& raw_indexes){

  // Open the pg_class and pg_attribute catalog tables
  Relation pg_class_rel = heap_open(RelationRelationId, AccessShareLock);
  Relation pg_attribute_rel = heap_open(AttributeRelationId, AccessShareLock);

  HeapScanDesc pg_class_scan = heap_beginscan_catalog(pg_class_rel, 0, NULL);

  // Go over all tuples in "pg_class"
  // pg_class has info about tables and everything else that has columns or is
  // otherwise similar to a table.
  // This includes indexes, sequences, views, composite types, and some kinds of
  // special relation.
  // So, each tuple can correspond to a table, index, etc.
  while (1) {
    // Get next tuple from pg_class
    HeapTuple pg_class_tuple = heap_getnext(pg_class_scan, ForwardScanDirection);

    if (!HeapTupleIsValid(pg_class_tuple)) break;

    Form_pg_class pg_class = (Form_pg_class)GETSTRUCT(pg_class_tuple);
    std::string relation_name = NameStr(pg_class->relname);
    char relation_kind = pg_class->relkind;

    // Handle only user-defined structures, not pg-catalog structures
    if (pg_class->relnamespace != PG_PUBLIC_NAMESPACE) continue;

    // TODO: Currently, we only handle relations and indexes
    if (pg_class->relkind != 'r' && pg_class->relkind != 'i') {
      continue;
    }

    // We only support tables with atleast one attribute
    int attnum = pg_class->relnatts;
    assert(attnum > 0);

    // Get the tuple oid
    // This can be a relation oid or index oid etc.
    Oid tuple_oid = HeapTupleHeaderGetOid(pg_class_tuple->t_data);
    std::vector<raw_column_info*> raw_columns = GetRawColumn(tuple_oid, pg_attribute_rel);

    switch (relation_kind) {
      case 'r':{
        raw_table_info* raw_table = GetRawTable(tuple_oid, relation_name, raw_columns);
        raw_tables.push_back(raw_table);
        }break;
      case 'i':{
        raw_index_info* raw_index = GetRawIndex(tuple_oid, relation_name, raw_columns);
        raw_indexes.push_back(raw_index);
        }break;
    }
    raw_columns.clear();
  }
  heap_endscan(pg_class_scan);
  heap_close(pg_attribute_rel, AccessShareLock);
  heap_close(pg_class_rel, AccessShareLock);
}

/**
 * @brief construct raw table with table oid, table name, raw columns
 * @param table oid
 * @param table name
 * @param raw columns
 * @return raw table
 */
raw_table_info* 
Bootstrap::GetRawTable(oid_t table_oid, std::string table_name, 
                       std::vector<raw_column_info*> raw_columns){

  raw_table_info* raw_table = (raw_table_info*) palloc(sizeof(raw_table_info));
  raw_table->table_oid = table_oid;
  raw_table->table_name = BootstrapUtils::CopyString(table_name.c_str());
  raw_table->raw_columns = (raw_column_info**) palloc(sizeof(raw_column_info*)*raw_columns.size());

  oid_t column_itr=0;
  for(auto raw_column : raw_columns){
    raw_table->raw_columns[column_itr++] = raw_column;
  }
  raw_table->column_count = column_itr;
  return raw_table;
}

/**
 * @brief construct raw index with index oid, index name, raw columns
 * @param index oid
 * @param index name
 * @param raw columns
 * @return raw index
 */
raw_index_info* 
Bootstrap::GetRawIndex(oid_t index_oid, 
                       std::string index_name,
                       std::vector<raw_column_info*> raw_columns){

  raw_index_info* raw_index = (raw_index_info*) palloc(sizeof(raw_index_info));

  Relation pg_index_rel;
  HeapScanDesc pg_index_scan;
  HeapTuple pg_index_tuple;

  pg_index_rel = heap_open(IndexRelationId, AccessShareLock);
  pg_index_scan = heap_beginscan_catalog(pg_index_rel, 0, NULL);

  // Go over the pg_index catalog table looking for indexes
  // that are associated with this table
  while (1) {
    Form_pg_index pg_index;

    pg_index_tuple = heap_getnext(pg_index_scan, ForwardScanDirection);
    if (!HeapTupleIsValid(pg_index_tuple)) break;

    pg_index = (Form_pg_index)GETSTRUCT(pg_index_tuple);

    // Search for the tuple in pg_index corresponding to our index
    if (pg_index->indexrelid == index_oid) {
      std::vector<std::string> key_column_names;

      for (auto raw_column : raw_columns) {
        key_column_names.push_back(raw_column->column_name);
      }

      IndexType method_type = INDEX_TYPE_BTREE;
      IndexConstraintType type;

      if (pg_index->indisprimary) {
        type = INDEX_CONSTRAINT_TYPE_PRIMARY_KEY;
      } else if (pg_index->indisunique) {
        type = INDEX_CONSTRAINT_TYPE_UNIQUE;
      } else {
        type = INDEX_CONSTRAINT_TYPE_DEFAULT;
      }

      // Store all index information here
      // This is required because we can only create indexes at once
      // after all tables are created
      // The order of table and index entries in pg_class table can be arbitrary

      raw_index->index_name = BootstrapUtils::CopyString(index_name.c_str());
      raw_index->index_oid = index_oid;
      raw_index->table_name = BootstrapUtils::CopyString(get_rel_name(pg_index->indrelid));
      raw_index->method_type = method_type;
      raw_index->constraint_type = type;
      raw_index->unique_keys = pg_index->indisunique;
      raw_index->key_column_names = BootstrapUtils::CopyStrings(key_column_names);
      raw_index->key_column_count = key_column_names.size();

      break;
    }
  }

  heap_endscan(pg_index_scan);
  heap_close(pg_index_rel, AccessShareLock);

  return raw_index;
}

/**
 * @brief construct raw column
 * @param tuple oid
 * @param pg_attribute_rel pg_attribute catalog relation 
 * @return raw columns
 */
std::vector<raw_column_info*> 
Bootstrap::GetRawColumn(Oid tuple_oid, Relation pg_attribute_rel) {

  HeapScanDesc pg_attribute_scan;
  HeapTuple pg_attribute_tuple;

  std::vector<raw_column_info*> raw_columns;

  // Scan the pg_attribute table for the relation oid we are interested in.
  pg_attribute_scan = heap_beginscan_catalog(pg_attribute_rel, 0, NULL);

  // Go over all attributes in "pg_attribute" looking for any entries
  // matching the given tuple oid.
  // For instance, this means the columns associated with a given relation oid.

  while (1) {
    Form_pg_attribute pg_attribute;

    // Get next <relation, attribute> tuple from pg_attribute table
    pg_attribute_tuple = heap_getnext(pg_attribute_scan, ForwardScanDirection);

    if (!HeapTupleIsValid(pg_attribute_tuple)) break;

    // Check the relation oid
    pg_attribute = (Form_pg_attribute)GETSTRUCT(pg_attribute_tuple);
    if (pg_attribute->attrelid == tuple_oid) {
      // Skip system columns in the attribute list
      if (strcmp(NameStr(pg_attribute->attname), "cmax") &&
          strcmp(NameStr(pg_attribute->attname), "cmin") &&
          strcmp(NameStr(pg_attribute->attname), "ctid") &&
          strcmp(NameStr(pg_attribute->attname), "xmax") &&
          strcmp(NameStr(pg_attribute->attname), "xmin") &&
          strcmp(NameStr(pg_attribute->attname), "tableoid")) {

        std::vector<raw_constraint_info*> raw_constraints;

        PostgresValueType postgresValueType = (PostgresValueType)pg_attribute->atttypid;
        ValueType value_type = PostgresValueTypeToPelotonValueType(postgresValueType);
        int column_length = pg_attribute->attlen;
        bool is_inlined = true;
        if (pg_attribute->attlen == -1) {
          column_length = pg_attribute->atttypmod;
          is_inlined = false;
        }

        // NOT NULL constraint
        if (pg_attribute->attnotnull) {
          raw_constraint_info* raw_constraint = (raw_constraint_info*)palloc(sizeof(raw_constraint_info));
          raw_constraint->constraint_type = CONSTRAINT_TYPE_NOTNULL;
          raw_constraint->constraint_name = ""; // TODO :: where is name?

          raw_constraints.push_back(raw_constraint);
        }

        // DEFAULT value constraint
        // TODO :: where is expression?
        if (pg_attribute->atthasdef) {
          raw_constraint_info* raw_constraint = (raw_constraint_info*)palloc(sizeof(raw_constraint_info));
          raw_constraint->constraint_type = CONSTRAINT_TYPE_DEFAULT;
          raw_constraint->constraint_name = ""; // TODO :: where is name?

          raw_constraints.push_back(raw_constraint);
 
        }

        raw_column_info* raw_column = (raw_column_info*)palloc(sizeof(raw_column_info));
        raw_column->column_type = value_type;
        raw_column->column_length = column_length;;
        raw_column->column_name = BootstrapUtils::CopyString(NameStr(pg_attribute->attname));
        raw_column->is_inlined = is_inlined;
        raw_column->raw_constraints = (raw_constraint_info**)palloc(sizeof(raw_constraint_info*)*raw_constraints.size());
        oid_t constraint_itr=0;
        for(auto raw_constraint : raw_constraints){
          raw_column->raw_constraints[constraint_itr++] = raw_constraint;
        }
        raw_column->constraint_count = constraint_itr;
        raw_constraints.clear();

        raw_columns.push_back(raw_column);
      }
    }
  }

  heap_endscan(pg_attribute_scan);

  return raw_columns;
}

void Bootstrap::GetRawForeignKeys(std::vector<raw_foreignkey_info*>& raw_foreignkeys) {
  Relation pg_constraint_rel;
  HeapScanDesc pg_constraint_scan;
  HeapTuple pg_constraint_tuple;

  oid_t database_oid = Bridge::GetCurrentDatabaseOid();
  assert(database_oid);

  pg_constraint_rel = heap_open(ConstraintRelationId, AccessShareLock);
  pg_constraint_scan = heap_beginscan_catalog(pg_constraint_rel, 0, NULL);

  // Go over the pg_constraint catalog table looking for foreign key constraints
  while (1) {
    Form_pg_constraint pg_constraint;

    pg_constraint_tuple =
        heap_getnext(pg_constraint_scan, ForwardScanDirection);
    if (!HeapTupleIsValid(pg_constraint_tuple)) break;

    pg_constraint = (Form_pg_constraint)GETSTRUCT(pg_constraint_tuple);

    // We only handle foreign key constraints here
    if (pg_constraint->contype != 'f') continue;

    // store raw information from here..

    raw_foreignkey_info* raw_foreignkey = (raw_foreignkey_info*) palloc(sizeof(raw_foreignkey_info));   

    // Extract oid
    raw_foreignkey->source_table_id = pg_constraint->conrelid;
    raw_foreignkey->sink_table_id = pg_constraint->confrelid;

    // Update/Delete action
    raw_foreignkey->update_action = pg_constraint->confupdtype;
    raw_foreignkey->delete_action = pg_constraint->confdeltype;

    // TODO :: Find better way..
    // column offsets, count
    bool isNull;
    Datum curr_datum =
        heap_getattr(pg_constraint_tuple, Anum_pg_constraint_conkey,
                     RelationGetDescr(pg_constraint_rel), &isNull);
    Datum ref_datum =
        heap_getattr(pg_constraint_tuple, Anum_pg_constraint_confkey,
                     RelationGetDescr(pg_constraint_rel), &isNull);

    ArrayType *curr_arr = DatumGetArrayTypeP(curr_datum);
    ArrayType *ref_arr = DatumGetArrayTypeP(ref_datum);
    int16 *curr_attnums = (int16 *)ARR_DATA_PTR(curr_arr);
    int16 *ref_attnums = (int16 *)ARR_DATA_PTR(ref_arr);
    int source_numkeys = ARR_DIMS(curr_arr)[0];
    int sink_numkeys = ARR_DIMS(ref_arr)[0];

    std::vector<int> source_column_offsets;
    std::vector<int> sink_column_offsets;

    // Populate foreign key column names
    for (int source_key_itr = 0; source_key_itr < source_numkeys;
         source_key_itr++) {
      AttrNumber attnum = curr_attnums[source_key_itr];
      source_column_offsets.push_back(attnum);
    }

    // Populate primary key column names
    for (int sink_key_itr = 0; sink_key_itr < sink_numkeys; sink_key_itr++) {
      AttrNumber attnum = ref_attnums[sink_key_itr];
      sink_column_offsets.push_back(attnum);
    }

    raw_foreignkey->source_column_offsets = (int*)palloc( sizeof(int)*source_column_offsets.size());
    raw_foreignkey->sink_column_offsets = (int*)palloc( sizeof(int)*sink_column_offsets.size());
    
    int column_itr=0;
    for(auto source_column_offset : source_column_offsets){
      raw_foreignkey->source_column_offsets[column_itr++] = source_column_offset;
    }
    raw_foreignkey->source_column_count = source_numkeys;

    column_itr=0;
    for(auto sink_column_offset : sink_column_offsets){
      raw_foreignkey->sink_column_offsets[column_itr++] = sink_column_offset;
    }
    raw_foreignkey->sink_column_count = sink_numkeys;

    std::string constraint_name = NameStr(pg_constraint->conname);

    raw_foreignkey->fk_name = BootstrapUtils::CopyString(constraint_name.c_str());
  }

  heap_endscan(pg_constraint_scan);
  heap_close(pg_constraint_rel, AccessShareLock);
}

void Bootstrap::CreateTables(raw_table_info** raw_tables, 
                             oid_t table_count){

  for(int table_itr=0; table_itr<table_count; table_itr++){
    auto raw_table = raw_tables[table_itr];
    auto columns = CreateColumns(raw_table->raw_columns,
                                 raw_table->column_count);
    
    bool status = DDLTable::CreateTable(raw_table->table_oid,
                                        raw_table->table_name,
                                        columns);
    if (status == true) {
      elog(LOG, "Create Table \"%s\" in Peloton", raw_table->table_name);
    } else {
      elog(ERROR, "Create Table \"%s\" in Peloton", raw_table->table_name);
    }
  }
}

void Bootstrap::CreateIndexes(raw_index_info** raw_indexes, 
                             oid_t index_count){

  for(int index_itr=0; index_itr<index_count; index_itr++){
    auto raw_index = raw_indexes[index_itr];
    auto key_column_names = CreateKeyColumnNames(raw_index->key_column_names, raw_index->key_column_count);

    IndexInfo index_info(raw_index->index_name,
                         raw_index->index_oid,
                         raw_index->table_name,
                         raw_index->method_type,
                         raw_index->constraint_type,
                         raw_index->unique_keys,
                         key_column_names);

    bool status = DDLIndex::CreateIndex(index_info);

    if (status == true) {
      elog(LOG, "Create Index \"%s\" in Peloton", raw_index->index_name);
    } else {
      elog(ERROR, "Create Index \"%s\" in Peloton", raw_index->index_name);
    }
  }
}

std::vector<catalog::Column>
Bootstrap::CreateColumns(raw_column_info** raw_columns, 
                         oid_t column_count){
  std::vector<catalog::Column> columns;

  for(int column_itr=0; column_itr<column_count; column_itr++){
    auto raw_column = raw_columns[column_itr];

    catalog::Column column( raw_column->column_type,
                            raw_column->column_length,
                            raw_column->column_name,
                            raw_column->is_inlined);
                            
    auto constraints = CreateConstraints(raw_column->raw_constraints,
                                         raw_column->constraint_count);
  
    for(auto constraint : constraints){
      column.AddConstraint(constraint);
    }
    columns.push_back(column);

  }
  return columns;
}

std::vector<std::string>
Bootstrap::CreateKeyColumnNames(char** raw_column_names, 
                                oid_t raw_column_count){
  std::vector<std::string> key_column_names;

  for(int column_itr=0; column_itr<raw_column_count; column_itr++){
    auto raw_column_name = raw_column_names[column_itr];

    key_column_names.push_back(raw_column_name);
  }
  return key_column_names;
}

std::vector<catalog::Constraint>
Bootstrap::CreateConstraints(raw_constraint_info** raw_constraints, 
                             oid_t constraint_count){
  std::vector<catalog::Constraint> constraints;

  for(int constraint_itr=0; constraint_itr<constraint_count; constraint_itr++){
    auto raw_constraint = raw_constraints[constraint_itr];

    catalog::Constraint constraint(raw_constraint->constraint_type,
                                   raw_constraint->constraint_name);

    constraints.push_back(constraint);
  }
  return constraints;
}

}  // namespace bridge
}  // namespace peloton
