/*==============================================================*/
/* Table: users                                                  */
/*==============================================================*/
create table users
(
   user_id           int not null auto_increment,
   user_name         text not null,
   user_password     text not null,
   user_permission   int not null,
   primary key (user_id)
);

insert into users(user_name, user_password, user_permission) values ("wps", "wps", 1);
insert into users(user_name, user_password, user_permission) values ("grj", "grj", 1);
insert into users(user_name, user_password, user_permission) values ("ljy", "ljy", 1);
insert into users(user_name, user_password, user_permission) values ("lzz", "lzz", 1);
insert into users(user_name, user_password, user_permission) values ("qlw", "qlw", 1);