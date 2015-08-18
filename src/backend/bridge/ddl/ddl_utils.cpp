//===----------------------------------------------------------------------===//
//
//                         PelotonDB
//
// ddl_utils.cpp
//
// Identification: src/backend/bridge/ddl/ddl_utils.cpp
//
// Copyright (c) 2015, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <iostream>

#include "backend/bridge/ddl/ddl_utils.h"
#include "backend/common/logger.h"
#include "backend/storage/database.h"

#include "parser/parse_type.h"
#include "utils/syscache.h"
#include "miscadmin.h"
#include "parser/parse_utilcmd.h"
#include "access/htup_details.h"
#include "utils/resowner.h"
#include "catalog/pg_type.h"
#include "commands/dbcommands.h"

namespace peloton {
namespace bridge {

//===--------------------------------------------------------------------===//
// DDL Utils
//===--------------------------------------------------------------------===//

/**
 * @brief preparing data
 * @param parsetree
 */
void DDLUtils::peloton_prepare_data(Node *parsetree) {
  switch (nodeTag(parsetree)) {
    case T_DropdbStmt: {
      DropdbStmt *stmt = (DropdbStmt *)parsetree;
      stmt->database_id = get_database_oid(stmt->dbname, stmt->missing_ok);
      break;
    }
    case T_CreateStmt:
    case T_CreateForeignTableStmt: {
      List *stmts = ((CreateStmt *)parsetree)->stmts;
      oid_t relation_oid = ((CreateStmt *)parsetree)->relation_id;
      ListCell *l;

      foreach (l, stmts) {
        Node *stmt = (Node *)lfirst(l);
        if (IsA(stmt, CreateStmt)) {
          CreateStmt *Cstmt = (CreateStmt *)stmt;

          // Get the column list from the create statement
          List *ColumnList = (List *)(Cstmt->tableElts);

          // Parse the CreateStmt and construct ColumnInfo
          ListCell *entry;
          int column_itr = 1;
          foreach (entry, ColumnList) {
            ColumnDef *coldef = static_cast<ColumnDef *>(lfirst(entry));

            Oid typeoid = typenameTypeId(NULL, coldef->typeName);
            int32 typemod;
            typenameTypeIdAndMod(NULL, coldef->typeName, &typeoid, &typemod);

            // Get type length
            Type tup = typeidType(typeoid);
            int typelen = typeLen(tup);
            ReleaseSysCache(tup);

            // For a fixed-size type, typlen is the number of bytes in the
            // internal
            // representation of the type. But for a variable-length type,
            // typlen
            // is
            // negative.
            if (typelen == -1) typelen = typemod;

            // Use existing TypeName structure
            coldef->typeName->type_oid = typeoid;
            coldef->typeName->type_len = typelen;

            if (coldef->raw_default != NULL || coldef->cooked_default != NULL)
              SetDefaultConstraint(coldef, column_itr++, relation_oid);
          }
        }
      }
      break;
    }
    default:
      // Don't need to prepare for other cases
      break;
      break;
  }
}

/**
 * @brief setting default constraint
 */
void DDLUtils::SetDefaultConstraint(ColumnDef *coldef, int column_itr,
                                    oid_t relation_oid) {
  Relation relation = heap_open(relation_oid, AccessShareLock);
  int num_defva = relation->rd_att->constr->num_defval;
  for (int def_itr = 0; def_itr < num_defva; def_itr++) {
    if (column_itr == relation->rd_att->constr->defval[def_itr].adnum) {
      char *default_expression =
          relation->rd_att->constr->defval[def_itr].adbin;
      coldef->cooked_default =
          static_cast<Node *>(stringToNode(default_expression));
    }
  }
  heap_close(relation, AccessShareLock);
}

/**
 * @brief parsing create statement
 * @param Cstmt a create statement
 * @param column_infos to create a table
 * @param refernce_table_infos to store reference table to the table
 */
void DDLUtils::ParsingCreateStmt(
    CreateStmt *Cstmt, std::vector<catalog::Column> &column_infos,
	__attribute__((unused))std::vector<catalog::ForeignKey> &foreign_keys) {
  assert(Cstmt);

  //===--------------------------------------------------------------------===//
  // Column Infomation
  //===--------------------------------------------------------------------===//

  // Get the column list from the create statement
  List *ColumnList = (List *)(Cstmt->tableElts);

  // Parse the CreateStmt and construct ColumnInfo
  ListCell *entry;
  foreach (entry, ColumnList) {
    ColumnDef *coldef = static_cast<ColumnDef *>(lfirst(entry));

    // Get the type oid and type mod with given typeName
    Oid typeoid = coldef->typeName->type_oid;
    int typelen = coldef->typeName->type_len;

    ValueType column_valueType =
        PostgresValueTypeToPelotonValueType((PostgresValueType)typeoid);
    int column_length = typelen;
    std::string column_name = coldef->colname;

    //===--------------------------------------------------------------------===//
    // Column Constraint
    //===--------------------------------------------------------------------===//

    std::vector<catalog::Constraint> column_constraints;

    if (coldef->constraints != NULL) {
      ListCell *constNodeEntry;

      foreach (constNodeEntry, coldef->constraints) {
        Constraint *ConstraintNode =
            static_cast<Constraint *>(lfirst(constNodeEntry));
        ConstraintType contype;
        std::string conname;

        // CONSTRAINT TYPE
        contype = PostgresConstraintTypeToPelotonConstraintType(
            (PostgresConstraintType)ConstraintNode->contype);

        // CONSTRAINT NAME
        if (ConstraintNode->conname != NULL) {
          conname = ConstraintNode->conname;
        } else {
          conname = "";
        }

        switch (contype) {
          case CONSTRAINT_TYPE_UNIQUE:
          case CONSTRAINT_TYPE_FOREIGN:
            continue;

          case CONSTRAINT_TYPE_NULL:
          case CONSTRAINT_TYPE_NOTNULL:
          case CONSTRAINT_TYPE_PRIMARY: {
            catalog::Constraint constraint(contype, conname);
            column_constraints.push_back(constraint);
            break;
          }
          case CONSTRAINT_TYPE_CHECK: {
            catalog::Constraint constraint(contype, conname,
                                           ConstraintNode->raw_expr);
            column_constraints.push_back(constraint);
            break;
          }
          case CONSTRAINT_TYPE_DEFAULT: {
            catalog::Constraint constraint(contype, conname,
                                           coldef->cooked_default);
            column_constraints.push_back(constraint);
            break;
          }
          default: {
            LOG_WARN("Unrecognized constraint type %d\n", (int)contype);
            break;
          }
        }
      }
    }  // end of parsing constraint

    catalog::Column column_info(column_valueType, column_length, column_name,
                                false);

    for (auto constraint : column_constraints)
      column_info.AddConstraint(constraint);

    // Insert column_info into ColumnInfos
    column_infos.push_back(column_info);
  }  // end of parsing column list
}

}  // namespace bridge
}  // namespace peloton
