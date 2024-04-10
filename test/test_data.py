import pandas as pd


def test_id():
    """
    Test that the id mapping is correct
    """
    # users.csv
    users_df = pd.read_csv("data/tomplay/users.csv")

    old_id, new_id = users_df["USER_ID"], users_df["u"]
    assert len(users_df) == len(new_id.unique())
    assert len(users_df) == len(old_id.unique())
    assert new_id.max() == len(new_id) - 1
    assert new_id.min() == 0
    user_id_mapping = dict(zip(old_id, new_id))

    # item.csv
    items_df = pd.read_csv("data/tomplay/items.csv")

    old_id, new_id = items_df["ITEM_ID"], items_df["i"]
    assert len(items_df) == len(new_id.unique())
    assert len(items_df) == len(old_id.unique())
    assert new_id.max() == len(new_id) - 1
    assert new_id.min() == 0
    item_id_mapping = dict(zip(old_id, new_id))

    # interactions.csv
    inter_df = pd.read_csv("data/tomplay/interactions.csv")

    old_user_id, old_item_id = inter_df["USER_ID"], inter_df["ITEM_ID"]
    new_user_id, new_item_id = old_user_id.map(
        user_id_mapping), old_item_id.map(item_id_mapping)
    assert (new_user_id == inter_df["u"]).all()
    assert (new_item_id == inter_df["i"]).all()
