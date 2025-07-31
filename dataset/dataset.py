import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict

class ActivityDataset(Dataset):
    def __init__(self, activity_chains, all_person_df, person_ids_df):
        """
        Args:
            activity_chains (np.ndarray): Numpy array of shape [N, 10, 5] ([N, time, feature]).
            all_person_df (pd.DataFrame): DataFrame with all person data from the survey.
            person_ids_df (pd.DataFrame): DataFrame from person_ids.csv, mapping chain index to HOUSEID/PERSONID.
        """
        self.activity_chains = activity_chains
        self.all_person_df = all_person_df.set_index(['HOUSEID', 'PERSONID'])
        self.person_ids_df = person_ids_df

        self.hh_members_features = [
            'DRIVER',       # 1. Driver's License Status
            'EDUC',         # 2. Education Level
            'OCCAT',        # 3. Job Category
            'R_SEX_IMP',    # 4. Gender
            'R_AGE_IMP',    # 5. Age
            'R_RACE',       # 6. Racial/Ethnic Identity
            'PTUSED',       # 7. PTUSED
            'R_RELAT',      # 8. Household Role
            'WORKER',       # 9. Employment Status
        ]
        self.full_features = [col for col in all_person_df.columns if col not in ['HOUSEID', 'PERSONID']]

        # Group personids by household for faster member lookup
        self.house_to_person_ids = all_person_df.groupby('HOUSEID')['PERSONID'].apply(list).to_dict()

    def __len__(self):
        return len(self.activity_chains)

    def __getitem__(self, idx):
        chain = torch.tensor(self.activity_chains[idx], dtype=torch.float)

        # Get the correct HOUSEID and PERSONID from the person_ids_df
        target_person_info = self.person_ids_df.iloc[idx]
        houseid = target_person_info['HOUSEID']
        personid = target_person_info['PERSONID']

        try:
            # Retrieve target person's features using the multi-index
            person_row = self.all_person_df.loc[(houseid, personid)]
            target_features = torch.tensor(person_row[self.full_features].values, dtype=torch.float)
        except KeyError:
            raise IndexError(f"Person with HOUSEID={houseid}, PERSONID={personid} not found in all_person_df.")

        # Get household members (excluding self)
        members = []
        if houseid in self.house_to_person_ids:
            for member_personid in self.house_to_person_ids[houseid]:
                if member_personid == personid:
                    continue
                try:
                    member_row = self.all_person_df.loc[(houseid, member_personid)]
                    partial_feat = member_row[self.hh_members_features].values
                    members.append(torch.tensor(partial_feat, dtype=torch.float))
                except KeyError:
                    # This member is in the household but not in the main dataframe, which is strange
                    # but we can skip them.
                    pass

        sample = {
            'activity_chain': chain,
            'target_features': target_features,
            'household_members': members,
            'label': chain[0, 0].long()
        }
        return sample