import sys
import time
from multiprocessing import Pool

from functools import partial

from create_description.utilities import clean


def relation_text_process(rel_str_list):
    """
    given a relation , which was transformed into a word vector
    """
    # rel_str = "/film/actor/film./film/performance/film, which is between /m/07nznf and /m/014lc_;/m/07nznf has a relationship of /award/award_winner/awards_won./award/award_honor/award with /m/02g3ft;/m/014lc_ has a relationship of /film/film/release_date_s./film/film_regional_release_date/film_release_region with /m/0f8l9c"
    relation_des_word_list = []

    for rel_str in rel_str_list:

        rel_str = rel_str.split(";")
        relation_mention = rel_str[0]
        relation_neighbours = rel_str[1:]

        relation_mention_list = relation_mention.split(" ")

        # print("relation_mention_list",relation_mention_list)
        relation_mention = clean(relation_mention_list[0])
        # print("relation_mention",relation_mention)

        two_entity = relation_mention_list[1:]

        beg = two_entity.index('between')
        end = two_entity.index('and')

        head_enti = two_entity[beg + 1:end]
        tail_enti = two_entity[end + 1:]

        relation_mention += ['which', 'is', 'between']

        head_enti_list = []
        if len(head_enti) == 1 and '/m/' in head_enti[0]:
            head_enti_list.append(head_enti[0])
        else:

            for z in range(len(head_enti)):
                head_enti_list += clean(head_enti[z])
        # tail_en = clean(())
        relation_mention += head_enti_list

        relation_mention += ['and']

        # print("relation_mention",relation_mention)

        tail_enti_list = []
        if len(tail_enti) == 1 and '/m/' in tail_enti[0]:
            tail_enti_list.append(tail_enti[0])
        else:

            for z in range(len(tail_enti)):
                tail_enti_list += clean(tail_enti[z])
        # tail_en = clean(())
        relation_mention += tail_enti_list

        # print("relation_mention",relation_mention)

        relation_description_list = []
        relation_description_list += relation_mention
        relation_description_list.append(".")

        neighbours_li = []
        #
        for i in range(len(relation_neighbours)):

            re_list = relation_neighbours[i].split(" ")  # 分解每个邻居
            # re_list = li[i].split(" ") # 分解每个邻居
            # re_list = li[i]

            if 'has' in re_list and 'with' in re_list:

                beg = re_list.index('has')
                end = re_list.index('with')

                # print(beg,end)
                sub_re_list = re_list[beg:end + 1]
                # print(sub_re_list)

                w_list = []
                # 加入头实体
                n_head = re_list[:beg]
                #
                n_head_list = []
                if len(n_head) == 1 and '/m/' in n_head[0]:
                    n_head_list.append(n_head[0])
                else:

                    for z in range(len(n_head)):
                        n_head_list += clean(n_head[z])
                # tail_en = clean(())
                head_en = n_head_list
                w_list += head_en  #

                # 处理关系
                for j in range(len(sub_re_list)):
                    w_list += clean(sub_re_list[j])

                # 处理尾实体
                n_tail = re_list[end + 1:]
                n_tail_list = []
                if len(n_tail) == 1 and '/m/' in n_tail[0]:
                    n_tail_list.append(n_tail[0])
                else:

                    for z in range(len(n_tail)):
                        n_tail_list += clean(n_tail[z])
                # tail_en = clean(())
                tail_en = n_tail_list

                w_list += tail_en  # 取尾巴实体
                # print(i)
                # w_list = clean(re_list)

                relation_description_list += w_list
                relation_description_list.append(".")

                neighbours_li.append(w_list)

                # print("w_list",w_list)

            else:

                sub_re_list = re_list
                # print(sub_re_list)
                # no_neighbour = clean(sub_re_list)
                relation_description_list += sub_re_list
                relation_description_list.append(".")

            # re_list = relation_neighbours[i].split(" ") # 分解每个邻居
            #
            # sub_re_list = re_list[1:-1]
            #
            # w_list = [re_list[0]] # 取头实体
            #
            # for j in range(len(sub_re_list)):
            #     w_list += clean(sub_re_list[j])
            #
            # w_list.append(re_list[-1]) # 取尾巴实体
            #
            # print(i,w_list)
            #
            # relation_description_list += w_list
            # relation_description_list.append(".")
        # print('relation_description_list',relation_description_list)
        relation_des_word_list.append(" ".join(relation_description_list))

    return relation_des_word_list


def relation_process(relation_name, head_obj, tail_obj, i):


    # print("==========hadoop data ==========\n")
    rel_des = str(relation_name[i])
              # + 'which is between ' + head_obj[i].label + ' and ' + tail_obj[
        # i].label + ';' \
        #       + head_obj[i].get_random_neighbour() + ';' + tail_obj[i].get_random_neighbour()
    #
    # rel_str = rel_des.split(";")

    relation_mention = rel_des + " ."

    # print("==========other data data ==========\n")
    # rel_des = str(relation_name[i]) + ' , ' + 'which is between ' + head_obj[i].label + ' and ' + tail_obj[
    #     i].label + ';' \
    #           + head_obj[i].get_random_neighbour() + ';' + tail_obj[i].get_random_neighbour()
    #
    # rel_str = rel_des.split(";")
    # relation_mention = rel_str[0]
    # relation_neighbours = rel_str[1:]
    # print("relation_neighbours",relation_neighbours)
    #
    # relation_mention_list = relation_mention.split(" ")
    # print("relation_mention_list",relation_mention_list)
    # relation_mention = clean(relation_mention_list[0])
    # print("relation_mention",relation_mention)
    #
    # two_entity = relation_mention_list[1:]
    #
    # # beg = two_entity.index('between')
    # # end = two_entity.index('and')
    # # head_enti = two_entity[beg+1:end]
    # # tail_enti = two_entity[end+1:]
    #
    # head_enti = head_obj[i].label.strip().split(" ")
    # tail_enti = tail_obj[i].label.strip().split(" ")
    #
    # relation_mention += ['which', 'is', 'between']
    #
    # head_enti_list = []
    # if len(head_enti) == 1 and '/m/' in head_enti[0]:
    #     head_enti_list.append(head_enti[0])
    # else:
    #
    #     for z in range(len(head_enti)):
    #
    #         head_enti_list += clean(head_enti[z])
    # # tail_en = clean(())
    # relation_mention += head_enti_list
    #
    # relation_mention += ['and']
    #
    # # print("relation_mention",relation_mention)
    #
    # tail_enti_list = []
    # if len(tail_enti) == 1 and '/m/' in tail_enti[0]:
    #     tail_enti_list.append(tail_enti[0])
    # else:
    #
    #     for z in range(len(tail_enti)):
    #
    #         tail_enti_list += clean(tail_enti[z])
    # # tail_en = clean(())
    # relation_mention += tail_enti_list
    #
    # # print("relation_mention",relation_mention)
    #
    # relation_description_list = []
    # relation_description_list += relation_mention
    # relation_description_list.append(".")
    #
    # neighbours_li = []
    # #
    # for i in range(len(relation_neighbours)):
    #
    #     re_list = relation_neighbours[i].split(" ") # 分解每个邻居
    #     # re_list = li[i].split(" ") # 分解每个邻居
    #     # re_list = li[i]
    #
    #     if 'has' in re_list and 'with' in re_list:
    #
    #         beg = re_list.index('has')
    #         end = re_list.index('with')
    #
    #         # print(beg,end)
    #         sub_re_list = re_list[beg:end+1]
    #         # print(sub_re_list)
    #
    #         w_list = []
    #         # 加入头实体
    #         n_head = re_list[:beg]
    #         #
    #         n_head_list = []
    #         if len(n_head) == 1 and '/m/' in n_head[0]:
    #             n_head_list.append(n_head[0])
    #         else:
    #
    #             for z in range(len(n_head)):
    #
    #                 n_head_list += clean(n_head[z])
    #         # tail_en = clean(())
    #         head_en = n_head_list
    #         w_list += head_en #
    #
    #
    #         # 处理关系
    #         for j in range(len(sub_re_list)):
    #
    #             w_list += clean(sub_re_list[j])
    #
    #         # 处理尾实体
    #         n_tail = re_list[end+1:]
    #         n_tail_list = []
    #         if len(n_tail) == 1 and '/m/' in n_tail[0]:
    #             n_tail_list.append(n_tail[0])
    #         else:
    #
    #             for z in range(len(n_tail)):
    #
    #                 n_tail_list += clean(n_tail[z])
    #         # tail_en = clean(())
    #         tail_en = n_tail_list
    #
    #         w_list += tail_en # 取尾巴实体
    #         # print(i)
    #         # w_list = clean(re_list)
    #
    #         relation_description_list += w_list
    #         relation_description_list.append(".")
    #         neighbours_li.append(w_list)
    #
    #     else:
    #
    #         sub_re_list = re_list
    #         # print(sub_re_list)
    #         # no_neighbour = clean(sub_re_list)
    #         relation_description_list += sub_re_list
    #         relation_description_list.append(".")

    #
    # return " ".join(relation_description_list)

    return relation_mention


def get_hrt_description_embedding(_h, _r, _t, entity_res, relation2id):
    """
    :param _h: head index
    :param _r: relation index
    :param _t: tail index
    :param ret: entity resource
    :param relation2id: relation id and name
    :return: the word embedding of _h,_r, and _t,

     entity_res = {'all_entity_obj_list': all_entity_obj_list,
                    'all_entity_description_word_list': all_entity_description_word_list,
                    'all_word_bag': all_word_bag,
                    'all_word_bag_dic': all_word_bag_dic,
                    'pre_trained_word_embedding': pre_trained_word_embedding
                    }

    """
    char = " "

    all_entity_res_obj = entity_res['all_entity_obj_list']
    all_entity_des_word = entity_res['all_entity_description_word_list']
    pre_word_embedding = entity_res['pre_trained_word_embedding']
    word_bag = entity_res['all_word_bag_dic']

    head_index = _h
    tail_index = _t
    relation_index = _r

    head_obj = [all_entity_res_obj[i] for i in head_index]
    tail_obj = [all_entity_res_obj[i] for i in tail_index]

    head_description_list = [" ".join(all_entity_des_word[i]) for i in head_index]  # get head entity description

    tail_description_list = [" ".join(all_entity_des_word[i]) for i in tail_index]  # get tail entity


    relation_name = relation2id[relation_index, 0]

    index_list = [i for i in range(len(relation_name))]

    pool = Pool(processes=1)

    pfunc = partial(relation_process, relation_name, head_obj, tail_obj)
    relation_description_word_list = pool.map(pfunc, index_list)

    pool.close()
    pool.join()

    # print("tail_description_list",tail_description_list,"\n")
    #
    #
    # for i in range(len(relation_name)):
    #
    #     rel_des = str(relation_name[i]) + ', ' + 'which is between ' + head_obj[i].label + ' and ' + tail_obj[
    #         i].label + ';' \
    #               + head_obj[i].get_random_neighbour() + ';' + tail_obj[i].get_random_neighbour()
    #     # print("rel_des ", rel_des)
    #     relation_description_list.append(rel_des)
    #
    # relation_description_word_list = relation_text_process(relation_description_list)

    # print("relation_description_word_list",relation_description_word_list)

    return head_description_list, relation_description_word_list, tail_description_list
