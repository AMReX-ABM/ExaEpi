/*! @file NAICS.H
 *  \brief Contains NAICS definitions */

#pragma once

#include <string>

using std::string;

/* Copied from UrbanPopAgentStruct.H for now to be used in CensusData
 * initializations and added an "uncategorized" code at the end; will
 * need to reconcile both at some point */

namespace NAICSCodes{
    /*! number of NAICS codes */
    const int NAICS_COUNT = 22;
    /*! string values for NAICS codes */
    static string naics_descriptions[NAICS_COUNT] = {"agr_ffh", "ext",     "utl",     "con", "mfg", "whl", "ret",
                                                     "trn_whs", "inf",     "fin_ins", "rrl", "prf", "mgt", "adm_wmr",
                                                     "edu",     "med_sca", "ent",     "afs", "srv", "pad", "wfh", "uncategorized"};
    /*! NAICS codes */
    struct NAICS {
        enum {
            agr_ffh,
            ext,
            utl,
            con,
            mfg,
            whl,
            ret,
            trn_whs,
            inf,
            fin_ins,
            rrl,
            prf,
            mgt,
            adm_wmr,
            edu,
            med_sca,
            ent,
            afs,
            srv,
            pad,
            wfh,
            uncat
        };
    };
}
